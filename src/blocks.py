"""基础组件占位：SEBlock, DoubleConv 等"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    对应论文 Section 3.1.2 及公式 (1), (2), (3)
    """

    def __init__(self, in_channels, reduction=16):
        """
        Args:
            in_channels (int): 输入特征图的通道数 (C)
            reduction (int): 降维比例 (Reduction Ratio)，用于控制 FC 层参数量
                             虽然论文未明确指定数值，但通常默认设为 16
        """
        super(SEBlock, self).__init__()

        # --- 1. Squeeze (挤压) ---
        # 对应公式 (1): z_c = F_sq(F_c) = Global Average Pooling
        # 将 HxW 的特征图压缩为 1x1 的点
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # --- 2. Excitation (激励) ---
        # 对应公式 (2): s = Sigma(W2 * ReLU(W1 * z))
        # 包含两个全连接层 (Fully Connected Layers)
        # 实际上在 CNN 中常用 kernel_size=1 的卷积来实现全连接层的功能，避免 shape 转换
        self.fc = nn.Sequential(
            # W1: 降维 (Channel -> Channel // reduction)
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            # Delta: ReLU 激活函数 [cite: 171]
            nn.ReLU(inplace=True),
            # W2: 升维 (Channel // reduction -> Channel)
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            # Sigma: Sigmoid 激活函数，输出 0~1 之间的权重 [cite: 171]
            nn.Sigmoid()
        )  # 现降维度，再升维度，降维度提取出不同通道的相关性情况，升维度回复通道，还能减少参数量

    def forward(self, x):
        """
        x: 输入特征图, 尺寸 [Batch, Channel, Height, Width]
        """
        b, c, _, _ = x.size()

        # --- Step 1: Squeeze ---
        # y shape: [b, c, 1, 1]
        y = self.avg_pool(x)

        # 为了过全连接层 (Linear)，需要展平成 [b, c]
        y = y.view(b, c)

        # --- Step 2: Excitation ---
        # y shape: [b, c] -> 得到通道权重向量 s
        y = self.fc(y)

        # 变回 [b, c, 1, 1] 以便与原始特征图相乘
        y = y.view(b, c, 1, 1)

        # --- Step 3: Scale (加权) ---
        # 对应公式 (3): F_scale = s * F
        # 利用广播机制 (Broadcasting)，将权重乘回原特征图
        return x * y

