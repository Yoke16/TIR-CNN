import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """CBAM: 通道注意力"""
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        reduction = max(1, in_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction, in_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=(2, 3), keepdim=True)
        maxv, _ = torch.max(x, dim=(2, 3), keepdim=True)
        out = self.mlp(avg) + self.mlp(maxv)
        out = self.sigmoid(out)
        return x * out


class SpatialAttention(nn.Module):
    """CBAM: 空间注意力"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        maxv, _ = torch.max(x, dim=1, keepdim=True)
        feat = torch.cat([avg, maxv], dim=1)
        out = self.conv(feat)
        out = self.sigmoid(out)
        return x * out


class CBAM(nn.Module):
    """CBAM = Channel + Spatial Attention"""
    def __init__(self, in_channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x


class ConvBlock(nn.Module):
    """U-Net 卷积块 + CBAM"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.cbam = CBAM(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.cbam(x)
        return x


class AttentionGate(nn.Module):
    """U-Net 解码阶段用的 Attention Gate，用于过滤 skip 特征"""
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # g: decoder feature, x: encoder feature
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=True)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        out = self.relu(g1 + x1)
        psi = self.psi(out)
        return x * psi


class TIR_CNN(nn.Module):
    """
    TIR-CNN（两阶段云反演模型）：
      - 输入:  16 通道 = [8 TIR] + [VZA] + [7 cloud params]
      - 输出1: CldType分类 (10类: 0-9)
      - 输出2: 7 通道云参数回归（已做 z-score 归一化）:
          0: DCOMP35_CPS
          1: DCOMP36_CPS
          2: DCOMP37_CPS
          3: DCOMP35_COD
          4: CldPressure
          5: CldHeight
          6: CldTemperature
    """

    def __init__(
        self,
        input_channels: int = 16,
        output_channels_cls: int = 10,  # CldType分类，10类
        output_channels_reg: int = 7,   # 7个云参数回归
        base_channels: int = 24,
        **kwargs,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels_cls = output_channels_cls
        self.output_channels_reg = output_channels_reg
        self.base_channels = base_channels

        # Encoder（共享特征提取）
        self.enc1 = ConvBlock(input_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.center = ConvBlock(base_channels * 8, base_channels * 16)

        # Decoder（共享解码器 + Attention Gate）
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.att4 = AttentionGate(base_channels * 16, base_channels * 8, base_channels * 8)
        self.dec4 = ConvBlock(base_channels * 16 + base_channels * 8, base_channels * 8)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.att3 = AttentionGate(base_channels * 8, base_channels * 4, base_channels * 4)
        self.dec3 = ConvBlock(base_channels * 8 + base_channels * 4, base_channels * 4)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.att2 = AttentionGate(base_channels * 4, base_channels * 2, base_channels * 2)
        self.dec2 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.att1 = AttentionGate(base_channels * 2, base_channels, base_channels)
        self.dec1 = ConvBlock(base_channels * 2 + base_channels, base_channels)

        # 两个输出头
        # 分类头：CldType (10类)
        self.classification_head = nn.Conv2d(base_channels, output_channels_cls, kernel_size=1)
        
        # 回归头：7个云参数
        self.regression_head = nn.Conv2d(base_channels, output_channels_reg, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        center = self.center(self.pool4(e4))

        # Decoder + Attention Gate
        up4 = self.up4(center)
        if up4.shape[2:] != e4.shape[2:]:
            up4 = F.interpolate(up4, size=e4.shape[2:], mode="bilinear", align_corners=True)
        e4_g = self.att4(up4, e4)
        d4 = self.dec4(torch.cat([up4, e4_g], dim=1))

        up3 = self.up3(d4)
        if up3.shape[2:] != e3.shape[2:]:
            up3 = F.interpolate(up3, size=e3.shape[2:], mode="bilinear", align_corners=True)
        e3_g = self.att3(up3, e3)
        d3 = self.dec3(torch.cat([up3, e3_g], dim=1))

        up2 = self.up2(d3)
        if up2.shape[2:] != e2.shape[2:]:
            up2 = F.interpolate(up2, size=e2.shape[2:], mode="bilinear", align_corners=True)
        e2_g = self.att2(up2, e2)
        d2 = self.dec2(torch.cat([up2, e2_g], dim=1))

        up1 = self.up1(d2)
        if up1.shape[2:] != e1.shape[2:]:
            up1 = F.interpolate(up1, size=e1.shape[2:], mode="bilinear", align_corners=True)
        e1_g = self.att1(up1, e1)
        d1 = self.dec1(torch.cat([up1, e1_g], dim=1))

        # 两阶段输出
        # 阶段1: CldType分类
        cldtype_logits = self.classification_head(d1)  # [B, 10, H, W]
        
        # 阶段2: 7个云参数回归
        regression_output = self.regression_head(d1)   # [B, 7, H, W]

        return cldtype_logits, regression_output
