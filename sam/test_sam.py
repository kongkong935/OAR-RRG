import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image

from sam.segment_anything import build_sam_vit_b, build_sam_vit_h
from sam.segment_anything.utils.transforms import ResizeLongestSide


class SAMInterface:
	"""
	统一的SAM接口：
	- encode_images: 可选，将原图编码为 image embeddings。
	- infer_from_tokens: 接收 image embeddings 与查询向量，返回最终掩码与 IoU 预测。

	输入：
	  - image_embeddings: [B, 256, Hf, Wf]（SAM image_encoder 的输出，或等价特征）
	  - query_embeddings: list[Tensor]，按 batch 切分的查询向量，每元素形如 [num_queries_i, 256]
	  - input_sizes: list[(H_in, W_in)] 对应每个样本的 SAM 输入尺寸（用于还原）
	  - original_sizes: list[(H_org, W_org)] 对应每个样本的原图尺寸
	输出：
	  - final_masks: list[Tensor]，每个样本对应 [num_queries_i, H_org, W_org]（默认取第一个候选）
	  - iou_pred: list[Tensor]，每个样本对应 [num_queries_i]
	"""

	def __init__(self, sam_checkpoint_path: str, device: str = "auto"):
		# 自动检测设备
		if device == "auto":
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = torch.device(device)

		
		# 初始化SAM模型并移动到设备
		self.sam = build_sam_vit_b(checkpoint=sam_checkpoint_path)
		self.sam.to(self.device)
		
		self.transform = ResizeLongestSide(1024)
		# self.transform = ResizeLongestSide(256)
		self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(self.device)
		self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(self.device)



	def infer_from_tokens(
		self,
		image_embeddings: torch.Tensor,
		query_embeddings: list,
		input_sizes: list,
		original_sizes: list,
		multimask_output: bool = False,
		enable_grad: bool = True,
	):
		"""
		从 query embeddings 推理分割掩码（支持梯度回流）
		
		Args:
			image_embeddings: [B, 256, Hf, Wf] - SAM image encoder 的输出
			query_embeddings: list of [N_seg_i, 256] - 每个样本的 query embeddings 列表
			input_sizes: list of (H, W) - SAM 输入尺寸
			original_sizes: list of (H, W) - 原始图像尺寸
			multimask_output: bool - 是否输出多个候选掩码
			enable_grad: bool - 是否启用梯度（训练时设为 True，推理时设为 False）
			
		Returns:
			final_masks_list: list of [N_seg_, H_org, W_org] - 每个样本的预测掩码列表
			iou_list: list of [N_seg_i] - 每个样本的 IoU 预测列表
		"""

		B = image_embeddings.size(0)
		device = next(self.sam.parameters()).device
		pe = self.sam.prompt_encoder.get_dense_pe()  # [256, Hf, Wf]
		return self._infer_from_tokens_impl(
			image_embeddings, query_embeddings, input_sizes, original_sizes,
			multimask_output, device, pe, B
		)


	
	def _infer_from_tokens_impl(
		self,
		image_embeddings: torch.Tensor,
		query_embeddings: list,
		input_sizes: list,
		original_sizes: list,
		multimask_output: bool,
		device: torch.device,
		pe: torch.Tensor,
		B: int,
	):
		"""内部实现方法，实际执行 SAM 解码逻辑（不使用 no_grad）"""
		final_masks_list = []
		iou_list = []

		for i in range(B):
			qi = query_embeddings[i].to(device)  # [N_seg_i, 256]
			if qi.numel() == 0:
				final_masks_list.append(torch.empty(0, device=device))
				iou_list.append(torch.empty(0, device=device))
				continue

			# ===== 一次性对该样本的所有 SEG query 做 batch 计算 =====
			# qi: [N_seg_i, 256] -> [N_seg_i, 1, 256] 作为 text_embeds
			text_embeds = qi.unsqueeze(1)

			# Prompt encoder（对 N_seg_i 个 query 并行编码）
			sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
				points=None,
				boxes=None,
				masks=None,
				text_embeds=text_embeds,  # [N_seg_i, 1, 256]
			)
			sparse_embeddings = sparse_embeddings.to(qi.dtype)  # [N_seg_i, 1, C]

			# image_embeddings 和 pe 仍保持单张图，[1, C, Hf, Wf]，
			# 让 SAM 内部在 predict_masks 中用 repeat_interleave 按 tokens.batch 方向展开
			image_embeddings_i = image_embeddings[i].unsqueeze(0)  # [1, 256, Hf, Wf]
			pe_i = pe  # [1, 256, Hf, Wf]

			# Mask decoder（对该样本的所有 SEG query 并行解码）
			low_res_masks, iou_pred = self.sam.mask_decoder(
				image_embeddings=image_embeddings_i,
				image_pe=pe_i,
				sparse_prompt_embeddings=sparse_embeddings,
				dense_prompt_embeddings=dense_embeddings,
				multimask_output=multimask_output,
			)  # low_res_masks: [N_seg_i, 1, H, W] 或 [N_seg_i, C, H, W]

			# Postprocess masks（对 N_seg_i 个掩码一起上采样）
			masks_up = self.sam.postprocess_masks(
				low_res_masks,
				input_size=input_sizes[i],
				original_size=original_sizes[i],
			)  # [N_seg_i, 1, H_org, W_org] 或 [N_seg_i, C, H_org, W_org]

			# 取每个 query 的第一个掩码: [N_seg_i, 1, H_org, W_org] -> [N_seg_i, H_org, W_org]
			sample_masks = masks_up[:, 0]  # [N_seg_i, H_org, W_org]
			sample_ious = iou_pred[:, 0] if iou_pred.numel() > 0 else torch.zeros(num_queries, device=device)

			final_masks_list.append(sample_masks)
			iou_list.append(sample_ious)

		return final_masks_list, iou_list



	def save_masks(self, masks: list, original_sizes: list, save_prefix: str = "mask"):
		"""
		保存掩码为图像文件（自动处理多查询向量）
		输入:
			- masks: 掩码列表
			- original_sizes: 原始图像尺寸列表
			- save_prefix: 保存文件名前缀（默认"mask"）
		输出:
			- 自动生成文件名，支持多个查询向量
		"""
		saved_files = []
		for i, (mask, orig_size) in enumerate(zip(masks, original_sizes)):
			# 确保掩码是numpy数组
			if isinstance(mask, torch.Tensor):
				mask = mask.cpu().numpy()

			# 处理多个掩码的情况
			if mask.ndim == 3 and mask.shape[0] > 1:
				# 多个掩码，分别保存
				for j in range(mask.shape[0]):
					single_mask = mask[j]
					save_path = f"{save_prefix}_{i+1}_query{j+1}.png"

					# 应用sigmoid并转换为0-255
					single_mask = 1 / (1 + np.exp(-single_mask))  # sigmoid
					single_mask = (single_mask * 255).astype(np.uint8)

					# 调整到原始尺寸
					mask_resized = cv2.resize(single_mask, (orig_size[1], orig_size[0]), interpolation=cv2.INTER_LINEAR)

					# 保存
					cv2.imwrite(save_path, mask_resized)
					saved_files.append(save_path)
					print(f"掩码已保存到: {save_path}")
			else:
				# 单个掩码
				if mask.ndim == 3:
					mask = mask.squeeze(0)  # 去掉单通道维度

				save_path = f"{save_prefix}_{i+1}_query1.png"

				# 应用sigmoid并转换为0-255
				mask = 1 / (1 + np.exp(-mask))  # sigmoid
				mask = (mask * 255).astype(np.uint8)

				# 调整到原始尺寸
				mask_resized = cv2.resize(mask, (orig_size[1], orig_size[0]), interpolation=cv2.INTER_LINEAR)

				# 保存
				cv2.imwrite(save_path, mask_resized)
				saved_files.append(save_path)
				print(f"掩码已保存到: {save_path}")

		return saved_files