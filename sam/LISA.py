from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

# 一些特殊token常量（与 LLaVA 的格式对齐）
from utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
)

# LLaVA 基座（语言模型与多模态封装）
from .llava.model.language_model.llava_llama import (
    LlavaLlamaForCausalLM,
    LlavaLlamaModel,
)

# SAM 构建器（ViT-H）
from .segment_anything import build_sam_vit_h


# -------------------------
# ------- 分割损失 --------
# -------------------------
def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 与原实现一致，用于数值稳定（可保持默认）
    eps=1e-6,
):
    """
    DICE 损失（对掩码的 IoU 风格度量）
    inputs:  [N, 1, H, W]  —— SAM 解码出的 logit
    targets: [N, 1, H, W]  —— GT 掩码 (0/1)
    num_masks: 该 batch 内的掩码数量（用于规范化）
    """
    inputs = inputs.sigmoid()               # 先把 logit 映射到 [0,1]
    inputs = inputs.flatten(1, 2)           # [N, H*W]（保持 batch 维度）
    targets = targets.flatten(1, 2)         # [N, H*W]

    # 标准 dice = 1 - (2 * 交集) / (两者像素和)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)

    # 跨所有掩码平均
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    二值交叉熵（对掩码的逐像素 BCE）
    inputs:  [N, 1, H, W] (logits)
    targets: [N, 1, H, W] (0/1)
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # 对每张掩码求均值，再对 batch 求和，最后用 num_masks 归一化
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


# --------------------------------
# ------- LISA 元模型封装 --------
# --------------------------------
class LisaMetaModel:
    """
    负责初始化与管理 LISA 的视觉侧（SAM）与
    文本→prompt 的投影层（text_hidden_fcs）。
    说明：在官方实现中该类实际混合继承自 LLaVA 的 Model，
    这里保持你贴出的结构。
    """
    def __init__(self, config, **kwargs):
        super(LisaMetaModel, self).__init__(config)

        self.config = config
        # 两种初始化路径：
        # 1) 首次构建时，外部通过 kwargs 指定 train_mask_decoder/out_dim/vision_pretrained
        # 2) 已在 config 中（多数来自 from_pretrained 的权重），则直接 initialize
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        # 1) 构建 SAM，并默认冻结所有参数
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False

        # 如需训练 mask decoder（推荐在医学域做轻微适配）
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # 2) 文本隐藏态 -> SAM prompt 的投影层（输出维度要与 SAM prompt 通道一致，通常 256）
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),  # out_dim 通常=256
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


# --------------------------------
# ------ LISA 顶层语言模型 -------
# --------------------------------
class LisaModel(LisaMetaModel, LlavaLlamaModel):
    """
    组合 LLaVA 的多模态 LLaMA 与上述 LISA 元模块。
    """
    def __init__(self, config, **kwargs):
        super(LisaModel, self).__init__(config, **kwargs)

        # 一些与 LLaVA 对齐的配置（图像特征选择/多模态 MLP 等）
        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class LISAForCausalLM(LlavaLlamaForCausalLM):
    """
    最外层可调用类：
    - 训练前向：model_forward()
    - 推理：evaluate() / generate() + 掩码解码
    """
    def __init__(self, config, **kwargs):
        # 补齐与 LLaVA 对齐的若干配置
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
            # 各项损失的权重（从外部传入）
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        else:
            config.mm_vision_tower = config.vision_tower

        # 关键：分割标记（如 [SEG_lung]）在词表中的 id
        self.seg_token_idx = kwargs.pop("seg_token_idx")

        # 先初始化 LLaVA 的父类
        super().__init__(config)

        # 再用 LisaModel 替换其内部 model（挂 SAM 与投影层）
        self.model = LisaModel(config, **kwargs)

        # 语言模型头（与 LLaVA 一致）
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 权重初始化
        self.post_init()

    # ----------- 视觉编码 -----------
    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        """
        走 SAM 的 image_encoder 把输入图像（已按 SAM 预处理）变成 image_embeddings。
        输入: pixel_values [B, 3, H_sam, W_sam]
        输出: image_embeddings [B, 256, H/4, W/4]（取决于 SAM 结构）
        """
        with torch.no_grad():  # 通常冻结 image_encoder
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    # ----------- 统一 forward 接口 -----------
    def forward(self, **kwargs):
        # 生成阶段（带 KV cache）直接走父类
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        # 训练走自定义前向
        return self.model_forward(**kwargs)

    # ----------- 训练/评估前向（带监督） -----------
    def model_forward(
        self,
        images: torch.FloatTensor,             # SAM 输入图（B,3,Hs,Ws）
        images_clip: torch.FloatTensor,        # CLIP 输入图（B,3,Hc,Wc）
        input_ids: torch.LongTensor,           # 模型的文本输入（含 <image> 和 [SEG_*]）
        labels: torch.LongTensor,              # 文本监督，非生成位置=-100
        attention_masks: torch.LongTensor,     # 注意力 mask
        offset: torch.LongTensor,              # 指示展开后的样本边界（用于多 seg 的索引切分）
        masks_list: List[torch.FloatTensor],   # 每张图的 GT 掩码列表（可能多器官）
        label_list: List[torch.Tensor],        # 每张图原始标签尺寸（用于还原）
        resize_list: List[tuple],              # SAM 输入前的 resize 尺寸
        inference: bool = False,               # True: 仅走前向拿隐藏态/不算文本 CE
        **kwargs,
    ):
        # (1) SAM 图像编码
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1

        # (2) 在 input_ids 中定位 [SEG_*] 的位置，构造一个 bool 掩码
        #    其 True 的位置对应我们要取隐藏态的 token
        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx     # 跳过首 token
        seg_token_mask = torch.cat(
            [seg_token_mask, torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda()],
            dim=1,
        )
        # trick：在前面补 255 个 False，抵消图像 token 在序列中的占位（假设只有1张图且在最前）
        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask],
            dim=1,
        )

        if inference:
            # 评估模式（不计算文本 CE；仅获取 hidden states）
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            # 将 CLIP 图像扩展到与文本序列同长（逐条输入）
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

            output_hidden_states = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                torch.cuda.empty_cache()

            # 拼回完整的 hidden_states（形状：[T, L, d]风格）
            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None

        else:
            # 训练模式：把每张图对应的 CLIP 图像复制到该样本包含的 token 数量
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)

            # LLaVA 前向，打开 hidden_states 与 labels（算文本 CE）
            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states

        # (3) 文本隐藏态 -> 256D prompt（只使用最后一层 hidden state）
        hidden_states = []
        assert len(self.model.text_hidden_fcs) == 1
        # output_hidden_states[-1] 形状约为 [B*T, L, d] 或 [B, L, d]（取决于上面拼接方式）
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))
        # 如果有多层分支就 stack+sum；此处只有一层
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)

        # (4) 仅在 seg_token 的位置取向量，得到每个 seg 的 prompt embedding
        pred_embeddings = last_hidden_state[seg_token_mask]     # [N_seg, 256]
        seg_token_counts = seg_token_mask.int().sum(-1)         # 每条样本中 seg token 的个数

        # (5) 用 offset 把展平的 pred_embeddings 切回每张图的列表
        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        )
        seg_token_offset = seg_token_offset[offset]  # 依据样本边界映射

        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])  # 每张图的多个 seg prompt
        pred_embeddings = pred_embeddings_

        # (6) SAM 解码（把上面的 256D prompt 当作 sparse prompt 喂给 prompt_encoder）
        multimask_output = False
        pred_masks = []
        for i in range(len(pred_embeddings)):
            # prompt_encoder 会返回 sparse/dense 两类 embedding
            (sparse_embeddings, dense_embeddings) = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1),  # [N_seg, 1, 256]
            )
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)

            # mask_decoder 输出低分辨率掩码（需 postprocess 到原图）
            low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

            # 还原到数据集中记录的尺寸（resize_list：SAM 输入前的尺寸；label_list[i].shape：GT 尺寸）
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks.append(pred_mask[:, 0])   # 取每个 seg 的第一张候选掩码

        model_output = output
        gt_masks = masks_list  # 与 pred_masks 一一对应（list of [N_seg,1,H,W]）

        if inference:
            # 评估/可视化阶段，直接返回掩码（不算文本 CE）
            return {"pred_masks": pred_masks, "gt_masks": gt_masks}

        # (7) 文本交叉熵损失（LLaVA 父类已计算）
        output = model_output.logits
        ce_loss = model_output.loss * self.ce_loss_weight

        # (8) 分割损失（逐图聚合）
        mask_bce_loss, mask_dice_loss, num_masks = 0, 0, 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]
            assert gt_mask.shape[0] == pred_mask.shape[0], \
                f"gt_mask.shape: {gt_mask.shape}, pred_mask.shape: {pred_mask.shape}"

            # 注意：这里 *gt_mask.shape[0] 是为了做“加权求和”，最后统一除以 num_masks
            mask_bce_loss += sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0]
            mask_dice_loss += dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0]
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        loss = ce_loss + mask_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }

    # ----------- 推理（文本生成 + 掩码解码） -----------
    def evaluate(
        self,
        images_clip,                 # CLIP 预处理图（B,3,Hc,Wc）
        images,                      # SAM 预处理图（B,3,Hs,Ws）
        input_ids,                   # 含 <image> 与 [SEG_*] 的提示
        resize_list,                 # SAM 前的尺寸（还原时用）
        original_size_list,          # 原图尺寸（还原时用）
        max_new_tokens=32,
        tokenizer=None,
    ):
        """
        先用 generate 生成文本（打开 output_hidden_states 以便拿到生成序列的隐藏态），
        再在生成的 token 序列中定位 [SEG_*] 的位置，取对应隐藏态，投影成 256D prompt，
        送入 SAM 解码得到掩码。
        返回：
          - output_ids: 生成序列 token id
          - pred_masks: list，每张图对应若干掩码 [N_seg, 1, H, W]
        """
        with torch.no_grad():
            # 1) 生成文本，并返回每一步的 hidden_states
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_hidden_states = outputs.hidden_states[-1]  # 取最后一层
            output_ids = outputs.sequences                   # 生成的 token 序列

            # 2) 在生成序列里找到 [SEG_*] 的位置
            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            seg_token_mask = torch.cat(
                [torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask],
                dim=1,
            )

            # 3) 文本隐藏态 -> 256D prompt
            hidden_states = []
            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))
            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]  # [N_seg, 256]

            # 4) 把展平的 seg prompt 切回每张图
            seg_token_counts = seg_token_mask.int().sum(-1)
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat([torch.zeros(1).long().cuda(), seg_token_offset], dim=0)

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            # 5) SAM 图像编码与解码
            image_embeddings = self.get_visual_embs(images)
            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                (sparse_embeddings, dense_embeddings) = self.model.visual_model.prompt_encoder(
                    points=None, boxes=None, masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )
                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                pred_masks.append(pred_mask[:, 0])

        return output_ids, pred_masks
