from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


# 基础 U-Net 组件 (解码器端保持原汁原味，用于特征的最终上采样融合)
class DoubleConv(nn.Sequential):
    """
    标准的连续两次 3x3 卷积提取特征。
    在解码器 (Decoder) 阶段，我们主要依靠它来平滑融合后的特征。
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class OutConv(nn.Sequential):
    """最后的 1x1 卷积，用于将通道数映射为我们要分类的类别数 (num_classes)"""

    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


# ===================================================================== #
# 论文创新点 1：通道多尺度模块 CMS (Channel Multi-scale module)
# 核心目的：打断残差网络中的特征复用，减少冗余信息，扩大有效感受野 (ERF)
# 部署位置：U-Net 的 Encoder (下采样/编码器) 阶段
# ===================================================================== #
class CMS_Block(nn.Module):
    def __init__(self, in_channels, out_channels, s=4):
        """
        :param s: 并行分支的数量，论文中默认为 4 个分支 (对应 1x1, 3x3, 5x5, 7x7)
        """
        super(CMS_Block, self).__init__()
        self.s = s

        # 1. Shortcut 残差连接：防止梯度消失
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        # 2. 降维操作：先用 1x1 卷积将特征图“挤压”并映射到输出通道
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 计算每个独立分支应该处理的通道数 (比如 64 / 4 = 16 通道)
        self.width = out_channels // s

        # 3. 核心创新：构建完全独立的多尺度并行分支
        self.branches = nn.ModuleList()
        for i in range(1, s + 1):
            k = 2 * i - 1  # 卷积核尺寸分别为: i=1->1x1, i=2->3x3, i=3->5x5, i=4->7x7

            if k == 1:
                # 尺度 1：只关注像素本身
                branch = nn.Sequential(
                    nn.Conv2d(self.width, self.width, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.width),
                    nn.ReLU(inplace=True)
                )
            elif k == 3:
                # 尺度 3：基础的局部感受野
                branch = nn.Sequential(
                    nn.Conv2d(self.width, self.width, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(self.width),
                    nn.ReLU(inplace=True)
                )
            else:
                # 尺度 5 和 7：宏观感受野，为了防止参数爆炸，这里严格使用 DW+PW 深度可分离卷积！
                # groups=self.width 代表 DW (Depthwise) 卷积，每个通道独立卷积
                branch = nn.Sequential(
                    nn.Conv2d(self.width, self.width, kernel_size=k, padding=k // 2, groups=self.width, bias=False),
                    nn.Conv2d(self.width, self.width, kernel_size=1, bias=False),  # PW (Pointwise) 卷积融合
                    nn.BatchNorm2d(self.width),
                    nn.ReLU(inplace=True)
                )
            self.branches.append(branch)

        # 4. 升维融合：将 4 个分支的结果拼接后，用 1x1 卷积再次融合通道信息
        self.conv2 = nn.Conv2d(self.width * s, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = self.shortcut(x)

        # 降维处理
        x = self.relu(self.bn1(self.conv1(x)))

        # 在通道维度 (dim=1) 上，将特征图切分成 s 份，送给不同的独立分支
        # 这是切断冗余特征流通的最关键一步！
        spx = torch.split(x, self.width, 1)

        out = []
        for i in range(self.s):
            out.append(self.branches[i](spx[i]))

        # 拼接独立提取的特征，并进行最后的融合
        out = torch.cat(out, dim=1)
        out = self.bn2(self.conv2(out))

        out += residual
        return self.relu(out)


class Down(nn.Sequential):
    """包含一次 2x2 下采样和一次 CMS 模块特征提取的 Encoder Block"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            CMS_Block(in_channels, out_channels)  # 使用 CMS 替换传统的 DoubleConv
        )



# ===================================================================== #
# 论文创新点 2：细节增强模块 DE 与多尺度嵌套 MSDE
# 核心目的：模拟人类中心-周边视觉，利用俄罗斯套娃式的感受野嵌套，凸显病灶模糊边界
# 部署位置：U-Net 的 Skip-connection (跳跃连接) 中间
# ===================================================================== #
class DE_Block(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.k_size = kernel_size

        if kernel_size == 1:
            self.eb = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        else:
            # 论文设定：k 是当前尺度，内层尺度为 k-2
            inner_k = kernel_size - 2

            # EB分支：(2i-1) x (2i-1) -> 即 k x k
            self.eb = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

            # EH分支：(2i-1) x [2(i-1)-1] -> 即 k x (k-2)
            # 这里的 padding 需要分别对应高和宽
            self.eh = nn.Conv2d(channels, channels, kernel_size=(kernel_size, inner_k),
                                padding=(kernel_size // 2, inner_k // 2), bias=False)

            # EV分支：[2(i-1)-1] x (2i-1) -> 即 (k-2) x k
            self.ev = nn.Conv2d(channels, channels, kernel_size=(inner_k, kernel_size),
                                padding=(inner_k // 2, kernel_size // 2), bias=False)

    def forward(self, x):
        if self.k_size == 1:
            return self.eb(x)

        x_eb = self.eb(x)
        x_eh = self.eh(x)
        x_ev = self.ev(x)

        return torch.cat([x_eb, x_eh, x_ev], dim=1)


class MSDE_Module(nn.Module):
    """
    多尺度细节增强模块 (Multi-scale Detail Enhanced Module)。
    通过并行 1,3,5,7 四个尺度，实现“中心嵌套”的弥散性对比 [cite: 275-278]。
    """

    def __init__(self, channels):
        super().__init__()

        # 并行实例化四个尺度的俄罗斯套娃演化模块
        self.de_1 = DE_Block(channels, kernel_size=1)
        self.de_3 = DE_Block(channels, kernel_size=3)
        self.de_5 = DE_Block(channels, kernel_size=5)
        self.de_7 = DE_Block(channels, kernel_size=7)

        # 计算合并后的总通道数：
        # de_1 只有 eb 1 个分支 -> 1 倍 channels
        # de_3, de_5, de_7 各有 eb, eh, ev 3 个分支 -> 各 3 倍 channels
        # 总计: 1 + 3 + 3 + 3 = 10 倍 channels
        concat_channels = channels * 10

        # 对应论文的 1x1 卷积重标定，将 10 倍膨胀的通道压缩回原始通道数，促使多尺度信息深度交融
        self.fusion = nn.Sequential(
            nn.Conv2d(concat_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 浅层特征图同时接受 4 个不同维度的放大镜审视 (完全并行，GPU 加速友好)
        out_1 = self.de_1(x)
        out_3 = self.de_3(x)
        out_5 = self.de_5(x)
        out_7 = self.de_7(x)

        # 对应论文 Eq(7): 跨尺度特征拼接 [cite: 265-266]
        out_concat = torch.cat([out_1, out_3, out_5, out_7], dim=1)

        # 特征降维输出，准备与解码器的深层特征融合
        return self.fusion(out_concat)


# ===================================================================== #
# 解码器层结构与 U-Net 整体拼装
# ===================================================================== #
class Up(nn.Module):
    """包含上采样、MSDE 跳跃连接强化以及融合特征提取的 Decoder Block"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        # 选择上采样方式：双线性插值或反卷积
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        # 重点：在这里实例化 MSDE 模块！
        # 跳跃连接处传来的特征图 (x2) 通道数刚好是 in_channels // 2
        self.msde = MSDE_Module(in_channels // 2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # 1. 把深层过来的特征图 x1 上采样放大
        x1 = self.up(x1)

        # 2. 尺寸对齐 (防止输入图片由于尺寸不是 2 的整数倍而产生的像素误差)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        # 3. 核心大招：在 x2 (浅层细节特征) 与 x1 拼接之前，
        # 让 x2 先通过 MSDE 模块洗礼，强化极其模糊的边界和病灶细节！
        x2 = self.msde(x2)

        # 4. 拼接 (Concat) 后通过标准双层卷积融合
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class MSDENet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2, bilinear: bool = True, base_c: int = 64):
        super(MSDENet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = CMS_Block(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)

        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)

        # 为解码器的每一层添加输出头，用于深层监督 (对应图3的 D1, D2, D3, D4 输出)
        self.out_conv1 = OutConv(base_c * 8 // factor, num_classes)
        self.out_conv2 = OutConv(base_c * 4 // factor, num_classes)
        self.out_conv3 = OutConv(base_c * 2 // factor, num_classes)
        self.out_conv4 = OutConv(base_c, num_classes)  # 最顶层主输出

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_shape = x.shape[2:]  # 记录输入图像的空间尺寸 (H, W)，用于后续深层监督输出的插值调整
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 深层监督输出 L3 (需要放大回原图尺寸)
        d1 = self.up1(x5, x4)
        out1 = self.out_conv1(d1)
        out1 = F.interpolate(out1, size=input_shape, mode='bilinear', align_corners=False)

        # 深层监督输出 L2 (需要放大回原图尺寸)
        d2 = self.up2(d1, x3)
        out2 = self.out_conv2(d2)
        out2 = F.interpolate(out2, size=input_shape, mode='bilinear', align_corners=False)

        # 深层监督输出 L1 (需要放大回原图尺寸)
        d3 = self.up3(d2, x2)
        out3 = self.out_conv3(d3)
        out3 = F.interpolate(out3, size=input_shape, mode='bilinear', align_corners=False)

        # 主预测输出 L0 (已经是原图尺寸，不需要插值)
        d4 = self.up4(d3, x1)
        out4 = self.out_conv4(d4)

        if self.training:
            return {"out": out4, "aux3": out3, "aux2": out2, "aux1": out1}
        else:
            return {"out": out4}