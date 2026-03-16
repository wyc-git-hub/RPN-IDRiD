# -*- coding: utf-8 -*-
"""
@Author: WYC
"""
from backbone import UNetBackbone
from peripheral import PeripheralVisionBranch
from central import *

class RPN(nn.Module):

    def __init__(self, PVB_LIST=None, CVB_LIST=None, n_channels=3, n_classes=1, bilinear=True):
        super(RPN, self).__init__()
        # 1. 初始化骨干网络
        self.backbone = UNetBackbone(n_channels, n_classes, bilinear)
        self.PVB_LIST = PVB_LIST
        self.CVB_LIST = CVB_LIST

        # 2. 定义 4 个周边视觉分支
        # 我们需要知道 Backbone 输出的 4 个特征图的通道数
        # 根据 models/backbone.py 的定义 (假设 bilinear=True):
        # F_d1 (Deepest):  256 channels
        # F_d2:            128 channels
        # F_d3:             64 channels
        # F_d4 (Shallowest):64 channels

        self.feature_channels = {'OUT_1': 256, 'OUT_2': 128, 'OUT_3': 64, 'OUT_4': 64}
        self.peripheral_branch_dict = nn.ModuleDict()  # 用于存储周边视觉分支，方便按名称访问

        for KEY, VALUE in self.feature_channels.items():
            self.peripheral_branch_dict.add_module(KEY, PeripheralVisionBranch(in_channels=VALUE))

        self.cvb_list = self.CVB_LIST.copy()
        # 反转列表，使得 CVB 从浅层到深层依次使用特征图
        # self.cvb_list.reverse()
        self.cvb_list_to = []
        for cvb in self.cvb_list:
            self.cvb_list_to.append(self.feature_channels[cvb])

        self.cvb = CentralVisionBranch(
            backbone_channels_list=self.cvb_list_to,
            feature_channels=self.cvb_list_to,
            cvb_channels=64
        )

    def forward(self, x):
        # --- 1. 骨干网络特征提取 ---
        # features 是一个列表: [f_d1, f_d2, f_d3, f_d4]
        # f_d1 最深 (语义强，尺寸小), f_d4 最浅 (细节多，尺寸大)
        features = self.backbone(x)

        # --- 2. 周边视觉分支预测 ---
        # 对每一层特征图，都生成一个 RSM
        rsms = []
        for pvb in self.PVB_LIST:
            rsms.append(self.peripheral_branch_dict[pvb](features[pvb]))

        pfm_logits = self.cvb(x, rsms[0], [features[i] for i in self.cvb_list]) # 这里假设 CVB 只使用第一个 RSM 作为引导
        # 返回 RSM 列表，pfm，用于在 Loss 计算时与 GT_RSM 列表一一对应
        return rsms, pfm_logits

