import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block
    论文提到的注意力机制，用于重新校准通道特征的重要性。
    
    结构:
    1. Squeeze: Global Average Pooling (全局平均池化)
    2. Excitation: FC -> ReLU -> FC -> Sigmoid
    3. Scale: 将生成的权重乘回原特征图
    """
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        # 为了保证降维层至少有1个通道，防止除以16后变成0
        mid_channels = max(in_channels // reduction, 1)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # 1. Squeeze: (B, C, H, W) -> (B, C, 1, 1)
        y = self.avg_pool(x).view(b, c)
        
        # 2. Excitation: (B, C) -> (B, C)
        y = self.fc(y).view(b, c, 1, 1)
        
        # 3. Scale: Channel-wise multiplication
        return x * y.expand_as(x)


class PeripheralVisionBranch(nn.Module):
    """
    周边视觉分支 (Peripheral Vision Branch)
    
    输入: 解码器的特征图 F_di (B, C, H, W)
    输出: 区域级监督掩模预测 RSM (B, 1, H, W)
    
    流程:
    F_di -> SE Block (特征增强) -> 1x1 Conv (通道压缩) -> Sigmoid (概率映射)
    """
    def __init__(self, in_channels):
        super(PeripheralVisionBranch, self).__init__()
        
        # 步骤 1: 特征增强 (SE Block)
        self.se_block = SEBlock(in_channels)
        
        # 步骤 2: 生成 RSM (1x1 Convolution)
        # 对应论文公式 (4): F_p = \sigma(f_{1x1}(SE(F_d)))
        self.conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
        
        # 步骤 3: 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 通过 SE Block 增强有效特征通道 (抑制背景噪音)
        x_se = self.se_block(x)
        
        # 2. 压缩为单通道热力图
        x_conv = self.conv1x1(x_se)
        
        # 3. 归一化到 [0, 1] 区间
        rsm_pred = self.sigmoid(x_conv)
        
        return rsm_pred
