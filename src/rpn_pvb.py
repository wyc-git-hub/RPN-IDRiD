import torch
import torch.nn as nn
from models.backbone import UNetBackbone
from models.peripheral import PeripheralVisionBranch
import torch

class RPN_Phase1(nn.Module):
    """
    RPN 第一阶段网络 (Phase 1 Network)
    
    仅包含:
    1. Backbone (U-Net Encoder-Decoder)
    def __init__(self, in_channels=3, features=[64, 128, 256]):
    - 训练网络能够从深到浅，在不同尺度上“发现”病灶区域。
    - 输出 4 个 RSM (Region-based Supervised Mask)。
        self.peripheral = PeripheralBranch(in_channels_list=features[::-1])
    """
    def __init__(self, n_channels=3, n_classes=1, bilinear=True, PVB_LIST = ['OUT_1', 'OUT_2', 'OUT_3', 'OUT_4'], CVB_LIST = ['OUT_1', 'OUT_2', 'OUT_3', 'OUT_4']):
        super(RPN_Phase1, self).__init__()
        # encoder returns features from shallow->deep; ensure peripheral gets channel sizes
        # convert feats to have channel sizes matching features[::-1]
        # Here we assume feats is [f1, f2, f3] with channels matching features
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
        
        # 为了代码的灵活性，这里显式定义通道列表
        # 如果你修改了 Backbone，请同步修改这里
        self.feature_channels = {'OUT_1':256, 'OUT_2':128, 'OUT_3':64, 'OUT_4':64}
        self.periph_branch_dict = nn.ModuleDict() # 用于存储周边视觉分支，方便按名称访问
        for (KEY, VALUE) in self.feature_channels:
            self.periph_branch_dict.add_module(KEY, PeripheralVisionBranch(in_channels=VALUE))

    def forward(self, x):
        # --- 1. 骨干网络特征提取 ---
        # features 是一个列表: [f_d1, f_d2, f_d3, f_d4]
        # f_d1 最深 (语义强，尺寸小), f_d4 最浅 (细节多，尺寸大)
        features = self.backbone(x)
        
        # --- 2. 周边视觉分支预测 ---
        # 对每一层特征图，都生成一个 RSM
        rsms = []
        for pvb in self.PVB_LIST:
            rsms.append(self.periph_branch_dict[pvb](features[pvb]))


        # 返回 RSM 列表，用于在 Loss 计算时与 GT_RSM 列表一一对应
        return rsms

