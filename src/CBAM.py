import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    CBAM 的第一步：通道注意力 (Channel Attention)
    相比 SE 模块，多了一个 MaxPool 分支，特征提取更全面。
    """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # 平均池化和最大池化是动态的，不需要定义层

        # 共享的全连接层 (Shared MLP)
        # 先降维 (in -> in/ratio)，再升维 (in/ratio -> in)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 平均池化分支
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # 2. 最大池化分支
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # 3. 结果相加后 Sigmoid
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    CBAM 的第二步：空间注意力 (Spatial Attention)
    专门解决 "病灶在哪里" 的问题。
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        # 卷积核大小通常用 7x7，感受野更大，能覆盖病灶周边
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        # 输入通道是 2 (因为一个是 MaxPool 得到的，一个是 AvgPool 得到的)
        # 输出通道是 1 (生成一张空间掩模)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 在通道维度上做 AvgPool -> [B, 1, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 2. 在通道维度上做 MaxPool -> [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # 3. 拼接 -> [B, 2, H, W]
        x = torch.cat([avg_out, max_out], dim=1)

        # 4. 卷积 + Sigmoid -> [B, 1, H, W]
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """
    完整的 CBAM 模块
    串联 Channel Attention 和 Spatial Attention
    """

    def __init__(self, planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # 先做通道注意力，加权原特征
        out = x * self.ca(x)
        # 再做空间注意力，再次加权
        result = out * self.sa(out)
        return result