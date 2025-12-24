import torch
import torch.nn as nn


class MultiTaskLogCoshLoss(nn.Module):
    """
    多任务回归损失，支持 NaN 掩膜：
      - preds/targets: [B, C, H, W]
      - targets 中的 NaN 自动忽略
      - 返回: total_loss, per_channel_losses[C]
    """

    def __init__(self, num_outputs: int = 7, weights=None):
        super().__init__()
        self.num_outputs = num_outputs
        if weights is None:
            weights = torch.ones(num_outputs, dtype=torch.float32)
        self.register_buffer("weights", weights)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        assert preds.shape == targets.shape, "preds / targets 形状必须一致"
        B, C, H, W = preds.shape
        assert C == self.num_outputs

        device = preds.device
        weights = self.weights.to(device)

        finite_mask = torch.isfinite(targets)  # [B, C, H, W]

        per_channel_losses = []
        total_loss = torch.tensor(0.0, device=device)

        for c in range(C):
            mask_c = finite_mask[:, c, :, :]
            if not torch.any(mask_c):
                loss_c = torch.tensor(0.0, device=device)
            else:
                diff = preds[:, c, :, :][mask_c] - targets[:, c, :, :][mask_c]
                loss_c = torch.log(torch.cosh(diff)).mean()
            total_loss = total_loss + weights[c] * loss_c
            per_channel_losses.append(loss_c)

        per_channel_losses = torch.stack(per_channel_losses)  # [C]
        return total_loss, per_channel_losses


class TwoStageLoss(nn.Module):
    """
    两阶段损失函数：
      - 阶段1: CldType分类损失 (CrossEntropyLoss, ignore_index=0)
      - 阶段2: 7个云参数回归损失 (MultiTaskLogCoshLoss)，只计算CldType>1的云区域
    
    参数:
        num_outputs: 回归输出通道数（默认7）
        reg_weights: 回归通道权重（可选）
        cls_weight: 分类损失权重（默认1.0）
        reg_weight: 回归损失权重（默认1.0）
    """

    def __init__(
        self,
        num_outputs: int = 7,
        reg_weights=None,
        cls_weight: float = 1.0,
        reg_weight: float = 1.0,
    ):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)  # 忽略背景类（0）
        self.reg_loss = MultiTaskLogCoshLoss(num_outputs=num_outputs, weights=reg_weights)
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight

    def forward(self, cldtype_logits, reg_preds, cldtype_target, reg_targets):
        """
        参数:
            cldtype_logits: [B, 10, H, W] - 分类logits
            reg_preds: [B, 7, H, W] - 回归预测（归一化空间）
            cldtype_target: [B, H, W] - 分类目标（0表示背景/无效像元）
            reg_targets: [B, 7, H, W] - 回归目标（归一化空间，NaN表示无效像元）
        
        返回:
            total_loss: 总损失
            cls_loss: 分类损失
            reg_loss: 回归损失
            per_channel: 每个回归通道的损失 [7]
        """
        # 1. 分类损失
        cls_loss = self.ce_loss(cldtype_logits, cldtype_target)

        # 2. 回归损失：只计算CldType>1的区域
        # 创建云区域mask（CldType>1）
        cloud_mask = cldtype_target > 1  # [B, H, W]

        # 将非云区域的回归目标设为NaN（让回归损失函数自动忽略）
        reg_targets_masked = reg_targets.clone()
        # 使用广播进行高效的向量化操作
        reg_targets_masked[:, :, ~cloud_mask] = float('nan')

        reg_loss, per_channel = self.reg_loss(reg_preds, reg_targets_masked)

        # 3. 总损失
        total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss

        return total_loss, cls_loss, reg_loss, per_channel
