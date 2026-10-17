"""
分割损失函数模块
- dice_loss: DICE损失（IoU风格度量）
- sigmoid_ce_loss: 二值交叉熵损失
- compute_segmentation_loss: 分割损失计算接口
"""

import torch
import torch.nn.functional as F
import numpy as np


@torch.no_grad()
def _auto_pos_weight(gt: torch.Tensor, clamp=(1.0, 20.0)):
    # 按正负比例自动给 BCE 设置 pos_weight（类极不平衡时有用）
    pos = gt.sum()
    neg = gt.numel() - pos
    w = neg / (pos + 1e-6)
    return w.clamp(*clamp)

def seg_loss_from_logits(
    pred_logits: torch.Tensor,   # [N, 512, 512]，原始logits（未sigmoid）
    gt_masks: torch.Tensor,      # [N, 512, 512]，0/1
    dice_w: float = 1.0,
    bce_w: float = 1.0,
    use_pos_weight: bool = False,
    eps: float = 1e-6,
):
    assert pred_logits.shape == gt_masks.shape and pred_logits.ndim == 3, \
        f"expect [N,512,512], got {pred_logits.shape} vs {gt_masks.shape}"

    # 对齐类型与设备
    logits = pred_logits.float()
    gt = gt_masks.to(dtype=logits.dtype, device=logits.device)

    # ---- Dice（基于概率）----
    prob = logits.sigmoid()                            # [N,512,512]
    p = prob.view(prob.size(0), -1)                    # [N, 262144]
    t = gt.view(gt.size(0), -1)
    inter = (p * t).sum(dim=1)
    card  = p.sum(dim=1) + t.sum(dim=1)
    dice = (2 * inter + eps) / (card + eps)            # [N]
    dice_loss = (1.0 - dice).mean()

    # ---- BCEWithLogits（基于logits）----
    # 扩一维做成 [N,1,H,W] 也行；直接 3D 也支持广播
    if use_pos_weight:
        pw = _auto_pos_weight(gt)
        bce_loss = F.binary_cross_entropy_with_logits(logits, gt, reduction="mean", pos_weight=pw)
    else:
        bce_loss = F.binary_cross_entropy_with_logits(logits, gt, reduction="mean")

    total = dice_w * dice_loss + bce_w * bce_loss
    return total, {"dice": dice_loss, "bce": bce_loss}

def dice_loss(inputs, targets, num_masks, scale=1000, eps=1e-6):
    inputs = inputs.sigmoid()               # 先把logit映射到[0,1]
    inputs = inputs.flatten(1, 2)           # [N, H*W]
    targets = targets.flatten(1, 2)         # [N, H*W]
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(inputs, targets, num_masks):
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


def compute_segmentation_loss(predicted_masks, masks_gt):
    mask_bce_loss_sum_weighted = 0.0
    mask_dice_loss_sum_weighted = 0.0
    total_masks = 0  # 记录整个 batch 中所有匹配掩码的总数
    for pred_mask, gt_mask_list in zip(predicted_masks, masks_gt):
        if pred_mask.numel() > 0 and len(gt_mask_list) > 0:
            gt_mask_tensor = torch.stack(gt_mask_list, dim=0)  # [N_gt, 512, 512]
            n_pred = pred_mask.shape[0]
            n_gt = gt_mask_tensor.shape[0]
            n_match = min(n_pred, n_gt)
            if n_match > 0:
                pred_input = pred_mask[:n_match].unsqueeze(1)
                gt_input = gt_mask_tensor[:n_match].unsqueeze(1)

                loss_bce_avg_i = sigmoid_ce_loss(pred_input, gt_input, num_masks=n_match)
                loss_dice_avg_i = dice_loss(pred_input, gt_input, num_masks=n_match)

                mask_bce_loss_sum_weighted += loss_bce_avg_i * n_match
                mask_dice_loss_sum_weighted += loss_dice_avg_i * n_match

                total_masks += n_match

    if total_masks > 0:
        loss_bce_final = mask_bce_loss_sum_weighted / ((total_masks + 1e-8)*100)
        loss_dice_final = mask_dice_loss_sum_weighted / ((total_masks + 1e-8)*100)

        return loss_bce_final , loss_dice_final

    else:
        device = predicted_masks[0].device if predicted_masks and len(predicted_masks) > 0 else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        return torch.tensor(0.0, device=device, requires_grad=True)



