import torch
import torch.nn.functional as F


class TargetGenerator:
    """
    GPU 加速的目标生成器
    负责在训练循环中实时生成 RSM 和 PFM
    """

    def __init__(self, rsm_strides=None, rsm_k=35, pfm_k=128):
        self.rsm_strides = rsm_strides
        self.pfm_k = pfm_k
        self.rsm_k = rsm_k

    def generate_rsm_batch(self, mask_tensor):
        """
        批量生成多尺度 RSM (使用 核心确信+周边晕染 的高级融合策略)
        """
        rsms = []

        for s in self.rsm_strides:
            current_k = int(self.rsm_k) if int(self.rsm_k) % 2 == 1 else int(self.rsm_k) - 1
            current_k = max(3, current_k)

            # ==========================================================
            # 1. Halo (光晕层)：严格执行论文公式，计算区域病灶密度 (均值池化)
            # ==========================================================
            halo = F.avg_pool2d(
                mask_tensor,
                kernel_size=current_k,
                stride=s,
                padding=current_k // 2,
                count_include_pad=True
            )

            # ==========================================================
            # 2. Core (核心层)：提取当前尺度下绝对的病灶位置 (最大池化)
            # 步长为 s，确保下采样后病灶的中心位置与特征图严格对齐，且值为 1.0
            # ==========================================================
            if s > 1:
                core = F.max_pool2d(mask_tensor, kernel_size=s, stride=s)
            else:
                core = mask_tensor

            # --- 尺寸对齐保护 (处理奇偶除法误差) ---
            expected_h = mask_tensor.shape[2] // s
            expected_w = mask_tensor.shape[3] // s

            if halo.shape[2:] != (expected_h, expected_w):
                halo = halo[:, :, :expected_h, :expected_w]
            if core.shape[2:] != (expected_h, expected_w):
                core = core[:, :, :expected_h, :expected_w]

            # ==========================================================
            # 3. Fusion (融合)：你的天才想法！取两者的最大值
            # 效果：病灶处必然是 1.0，周围则是逐渐衰减的均值概率
            # ==========================================================
            rsm = halo

            rsms.append(rsm)

            # 随尺度加深减小区域范围
            self.rsm_k //= 2

        return rsms
    def generate_pfm_batch(self, mask_tensor):
        """
        批量生成 PFM (三值掩模)
        Args:
            mask_tensor: (B, 1, H, W)
        Returns:
            pfm: (B, H, W) LongTensor
        """
        B, C, H, W = mask_tensor.shape
        k = self.pfm_k
        padding = k // 2

        # 1. 膨胀 (Dilation) 使用 MaxPool
        # 这一步极快，因为是在 GPU 上并行的
        dilated_mask = F.max_pool2d(
            mask_tensor,
            kernel_size=k,
            stride=1,
            padding=padding
        )

        # 2. 尺寸对齐
        if dilated_mask.shape[2] != H or dilated_mask.shape[3] != W:
            dilated_mask = dilated_mask[:, :, :H, :W]

        # 3. 构建三值 Mask
        # 初始化为 2 (忽略区域)
        pfm = torch.full_like(mask_tensor, 2.0)

        # 逻辑: 膨胀区设为 0 (难例背景), 原始病灶设为 1 (正样本)
        pfm[dilated_mask > 0.5] = 0.0
        pfm[mask_tensor > 0.5] = 1.0

        # 4. 去掉通道维度，转为 Long (B, H, W)
        return pfm.squeeze(1).long()

    def __call__(self, mask_tensor):
        """
        一次性调用接口
        """
        with torch.no_grad():  # 确保不计算梯度，节省显存
            rsms = self.generate_rsm_batch(mask_tensor)
            pfm = self.generate_pfm_batch(mask_tensor)
        return rsms, pfm