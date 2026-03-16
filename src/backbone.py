import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    基础卷积模块：(Conv3x3 -> BN -> ReLU) * 2
    这是 U-Net 的标准构建块。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    下采样模块：MaxPool2d -> DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    上采样模块：Upsample -> Concat -> DoubleConv
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # 根据显存情况选择双线性插值或转置卷积，这里默认使用双线性插值以减少棋盘格效应
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # 拼接后的通道数 = 输入通道数的一半 + Skip Connection 的通道数
            # 通常 in_channels 是拼接后的总数，所以卷积层负责减半通道
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: 来自上一层解码器的输入 (需要上采样)
        x2: 来自编码器对应的 Skip Connection (x2 尺寸 >= x1)
        """
        x1 = self.up(x1)
        
        # 处理输入尺寸可能因为 padding 导致的不匹配问题
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # 拼接 (Channel 维度)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetBackbone(nn.Module):
    """
    RPN 的主干网络 (基于 U-Net)
    
    结构特点：
    - 5 层 Encoder (Feature Extraction)
    - 4 层 Decoder (Feature Reconstruction)
    - 返回 4 个不同尺度的解码特征图 F_di，用于周边视觉分支
    """
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(UNetBackbone, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # --- Encoder (编码器) ---
        # Input: (B, 3, 640, 640)
        self.inc = DoubleConv(n_channels, 64)       # -> (B, 64, 640, 640)
        self.down1 = Down(64, 128)                  # -> (B, 128, 320, 320)
        self.down2 = Down(128, 256)                 # -> (B, 256, 160, 160)
        self.down3 = Down(256, 512)                 # -> (B, 512, 80, 80)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)      # -> (B, 512, 40, 40) (If bilinear, to save memory)
        # 概率 p=0.3 到 0.5 之间最合适，每次前向传播随机“丢弃” 30% 的神经元
        self.dropout_bottle = nn.Dropout(p=0.3)
        self.dropout_up1 = nn.Dropout(p=0.3)
        # --- Decoder (解码器) ---
        # 这里的 in_channels 计算包含了 Skip Connection 的拼接
        self.up1 = Up(1024, 512 // factor, bilinear) # -> F_d1 (B, 256, 80, 80)
        self.up2 = Up(512, 256 // factor, bilinear)  # -> F_d2 (B, 128, 160, 160)
        self.up3 = Up(256, 128 // factor, bilinear)  # -> F_d3 (B, 64, 320, 320)
        self.up4 = Up(128, 64, bilinear) # -> F_d4 (B, 64, 640, 640)

    def forward(self, x):
        # --- Encoder Path ---
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4) # Bottleneck
        # --- Decoder Path ---
        # 这里的输出 f_d1 ~ f_d4 将被送入 Peripheral Vision Branch
        
        # Layer 1 (Deepest, Stride=8 if input 640, output 80)
        # 注意：这里具体的尺寸取决于 input size 和 bilinear 设置
        f_d1 = self.up1(x5, x4)
        # Layer 2
        f_d2 = self.up2(f_d1, x3)
        
        # Layer 3
        f_d3 = self.up3(f_d2, x2)
        
        # Layer 4 (Shallowest, Output size == Input size)
        f_d4 = self.up4(f_d3, x1)
        
        # 返回解码器的特征图列表，顺序从深到浅：[Deep, ..., Shallow]
        # 对应 Config.RSM_STRIDES 的 [32, 16, 8, 4] 逻辑 (具体倍率视 maxpool 次数而定)
        # x5 (stride 16) -> up -> f_d1 (stride 8)
        # f_d1 (stride 8) -> up -> f_d2 (stride 4) ... 
        # 修正：根据标准 U-Net 5层结构：
        # x1: 1, x2: 1/2, x3: 1/4, x4: 1/8, x5: 1/16 (如果 down4 是最后一层)
        # 实际上论文中通常指：
        # Deepest feature for RSM1
        
        return {'OUT_1':f_d1, 'OUT_2':f_d2, 'OUT_3':f_d3, 'OUT_4':f_d4}

