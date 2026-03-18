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
        self.rsm_k_u = rsm_k

    def generate_rsm_batch(self, mask_tensor):
        """
        批量生成多尺度 RSM (使用 核心确信+周边晕染 的高级融合策略)
        """
        rsms = []
        rsm_k = self.rsm_k_u  # 每次调用时重置 rsm_k，确保多尺度生成时正确递减
        for s in self.rsm_strides:
            current_k = int(rsm_k) if int(rsm_k) % 2 == 1 else int(rsm_k) - 1
            current_k = max(3, current_k)

            halo = F.avg_pool2d(
                mask_tensor,
                kernel_size=current_k,
                stride=s,
                padding=current_k // 2,
                count_include_pad=True
            )

            expected_h = mask_tensor.shape[2] // s
            expected_w = mask_tensor.shape[3] // s

            if halo.shape[2:] != (expected_h, expected_w):
                halo = halo[:, :, :expected_h, :expected_w]

            rsms.append(halo)

            # 随尺度加深减小区域范围
            rsm_k //= 2

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