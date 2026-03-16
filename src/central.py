import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import SEBlock


class CentralConvBlock(nn.Module):
    """
    中央视觉分支专用的卷积块
    """

    def __init__(self, in_channels, out_channels):
        super(CentralConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class CentralVisionBranch(nn.Module):
    """
    中央视觉分支 (Central Vision Branch) - 修复通道不匹配版
    """

    def __init__(self, backbone_channels_list, feature_channels=None, cvb_channels=64):
        """
        Args:
            backbone_channels_list: Backbone Decoder 层输出通道列表 [64, 64, 128...]
            feature_channels: (通常与 backbone_channels_list 相同，保留参数接口)
            cvb_channels: 中央分支内部维持的基础通道数 (64)
        """
        super(CentralVisionBranch, self).__init__()
        self.cvb_channels = cvb_channels

        # 1. 初始处理层
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, cvb_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cvb_channels),
            nn.ReLU(inplace=True)
        )

        # 2. 动态构建融合阶段
        self.stages = nn.ModuleList()
        self.se_blocks = nn.ModuleList()

        # 初始输入通道数 (来自 init_conv)
        current_in_c = cvb_channels

        # 遍历每一层进行融合
        # backbone_channels_list 里的 bb_dim 就是要拼接进来的特征通道数
        for i, bb_dim in enumerate(backbone_channels_list):
            # [Step A] CentralConvBlock
            # 作用：处理上一层的特征，并将通道数压缩回 cvb_channels (64)
            # 这样可以防止随着层数增加通道数无限膨胀，同时起到特征整合作用
            self.stages.append(CentralConvBlock(current_in_c, cvb_channels))

            # [Step B] SEBlock
            # 在 forward 中，ConvBlock 输出 (64) 会和 Backbone 特征 (bb_dim) 拼接
            # 所以 SEBlock 需要接受的通道数是: 64 + bb_dim
            concat_channels = cvb_channels + bb_dim
            self.se_blocks.append(SEBlock(concat_channels))

            # [Step C] 更新下一层的输入通道数
            current_in_c = concat_channels

        # 3. 最终预测层
        # 输入是最后一个 SE Block 的输出 (current_in_c)
        self.final_conv = nn.Sequential(
            nn.Conv2d(current_in_c, current_in_c // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(current_in_c // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(current_in_c // 2, current_in_c // 4, kernel_size=1),
            nn.BatchNorm2d(current_in_c // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(current_in_c // 4, 1, kernel_size=1)
        )

    def forward(self, img, rsm_pred, backbone_features_list):
        # 1. Attention Masking
        if rsm_pred.shape[2:] != img.shape[2:]:
            rsm_up = F.interpolate(rsm_pred, size=img.shape[2:], mode='bilinear', align_corners=True)
        else:
            rsm_up = rsm_pred

        weighted_img = img * rsm_up

        # 2. Initial Conv
        x = self.init_conv(weighted_img)  # -> [B, 64, H, W]

        # 3. Cascade Fusion Loop
        for i, (stage_conv, se_block) in enumerate(zip(self.stages, self.se_blocks)):
            # A. 特征整合与压缩 (Input: current_in_c -> Output: 64)
            x = stage_conv(x)

            # B. 融合 Backbone 特征
            if i < len(backbone_features_list):
                bb_feat = backbone_features_list[i]

                # 尺寸对齐
                if bb_feat.shape[2:] != x.shape[2:]:
                    bb_feat = F.interpolate(bb_feat, size=x.shape[2:], mode='bilinear', align_corners=True)

                # 拼接: [B, 64, H, W] cat [B, bb_dim, H, W] -> [B, 64+bb_dim, H, W]
                x = torch.cat([x, bb_feat], dim=1)

            # C. SE Block 增强 (Input: 64+bb_dim)
            x = se_block(x)

        # 4. Final Prediction
        out = self.final_conv(x)
        return out