import torch
from torch import nn
import torch.nn.functional as F
import train_utils.distributed_utils as utils
from .dice_coefficient_loss import dice_loss, build_target
import sys
import os

# 将项目根目录加入环境，以便导入 target_generators
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_utils.target_generators import TargetGenerator


def criterion(pred_rsms, pred_pfm, gt_rsms, gt_pfm, ignore_mask):
    """
    针对 RPN 架构自定义的联合 Loss 计算。
    包含：周边视觉分支的多个 RSM Loss + 中央视觉分支的 PFM Loss。
    """
    loss = 0.0

    # --- 1. 计算 PVB 损失 (多尺度 RSM 监督) ---
    neg_weight_val = 1.0  # 纯背景的基础权重
    min_pos_weight = 2.0  # 只要有病灶，最低保底权重 (防止 MA 小病灶被忽略)
    density_scale = 1.0  # 密度带来的额外权重加成 (密度为1时，最高权重为 5 + 5 = 10)

    for pred_rsm, gt_rsm in zip(pred_rsms, gt_rsms):
        # 统一 ignore_mask 的尺寸
        if pred_rsm.shape[2:] != ignore_mask.shape[2:]:
            curr_ignore = F.interpolate(ignore_mask.float(), size=pred_rsm.shape[2:], mode='nearest').bool()
        else:
            curr_ignore = ignore_mask

        # 反转得到有效区域 (去除 255 对应的眼球外区域)
        valid_rsm = ~curr_ignore

        if valid_rsm.sum() > 0:
            pred_valid = pred_rsm[valid_rsm]
            gt_valid = gt_rsm.float()[valid_rsm]

            # ==========================================
            # 高阶修改：基于密度的动态权重 (Density-Aware Weighting)
            # ==========================================
            # 1. 初始化全为背景权重 (1.0)
            weight_tensor = torch.full_like(gt_valid, neg_weight_val)

            # 2. 找到所有包含病灶的区域 (GT > 0)
            pos_mask = gt_valid > 0

            if pos_mask.sum() > 0:
                # 3. 核心公式：保底权重 + 密度 * 缩放系数
                # 假设 gt_valid=0.01 (小病灶): weight = 5.0 + 0.01*5.0 = 5.05 (靠保底存活)
                # 假设 gt_valid=1.00 (大病灶): weight = 5.0 + 1.00*5.0 = 10.0 (获得满额重视)
                weight_tensor[pos_mask] = min_pos_weight + gt_valid[pos_mask] * density_scale

            # 将构建好的动态权重张量传入 weight 参数
            loss += F.binary_cross_entropy(
                pred_valid,
                gt_valid,
                weight=weight_tensor
            )

    # --- 2. 计算 CVB 损失 (全尺寸 PFM 监督) ---
    # gt_pfm 中 2 为光晕过渡带(不参与训练)，ignore_mask 对应眼球外死黑区
    valid_mask_pfm = (gt_pfm != 2) & (~ignore_mask.squeeze(1))

    if valid_mask_pfm.sum() > 0:
        pred_valid = pred_pfm.squeeze(1)[valid_mask_pfm]
        gt_valid = gt_pfm[valid_mask_pfm].float()

        # 中央分支输出的是 logits，所以用带有 logits 的 BCE，数值更稳定
        bce_loss = F.binary_cross_entropy_with_logits(pred_valid, gt_valid)
        loss += bce_loss

    return loss


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(2)
    dice = utils.DiceCoefficient(num_classes=2, ignore_index=255)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # 收集所有的概率和标签张量（保存在 GPU 上）
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 1, header):
            image, target = image.to(device), target.to(device)

            target_long = target.long()
            if target_long.dim() == 4:
                target_long = target_long.squeeze(1)

            # 获取 PVB 和 CVB 的预测
            pred_rsms, pred_pfm = model(image)

            # 1. 获取 CVB 的细节预测概率
            cvb_prob = torch.sigmoid(pred_pfm).squeeze(1)

            # 2. 获取 PVB 的大局区域预测概率
            rsm_prob = pred_rsms[0].squeeze(1)
            if rsm_prob.shape != cvb_prob.shape:
                rsm_prob = F.interpolate(rsm_prob.unsqueeze(1), size=cvb_prob.shape[-2:], mode='bilinear',
                                         align_corners=True).squeeze(1)

            # ==========================================
            # 核心修复：使用“掩码乘法”代替“概率直乘” (相当于连续域的逻辑与)
            # ==========================================
            # 因为 PVB 偏灰白（宁滥勿缺），我们给它一个较低的宽容阈值（比如 0.1 或 0.15）
            # 这代表只要 PVB 觉得这块区域"哪怕有一点点嫌疑"，我们就允许 CVB 去发挥
            rsm_mask = (rsm_prob > 0.15).float()

            # 逻辑与：掩码外直接归零（抑制背景），掩码内 100% 保留 CVB 原本的自信度（不发生衰减）
            output_prob = cvb_prob * rsm_mask
            # ==========================================

            # 过滤掉标签为 255 的眼球外死黑区域
            valid_mask = target_long != 255

            # 直接提取有效区域并存入列表 (无需转 cpu 或 numpy)
            all_probs.append(output_prob[valid_mask])
            all_targets.append(target_long[valid_mask].float())

            # 二分类阈值切分
            pred_label = (output_prob > 0.5).long()

            # --- 关键修复：补回维度以适配原框架的 dice.update ---
            # 原框架内部会做 pred.argmax(dim=1)，因此我们需要提供一个 4 维张量 [B, C, H, W]
            # 对于二分类，我们可以构造一个包含 [背景概率, 前景概率] 的伪张量
            # 或者更简单的做法：直接给 pred_label 增加一个维度，并确保它能过 argmax

            # 构造一个符合原框架要求的“伪概率”张量 [B, 2, H, W]
            # 通道 0 代表背景，通道 1 代表病灶
            pred_for_dice = torch.stack([1 - output_prob, output_prob], dim=1)

            # 更新混淆矩阵（保持不变）
            confmat.update(target_long.flatten(), pred_label.flatten())

            # 使用构造的 [B, 2, H, W] 张量更新 Dice
            dice.update(pred_for_dice, target_long)

        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()

        # ==========================================
        # 纯 PyTorch 实现 AUCPR (Average Precision)
        # ==========================================
        # 将所有 batch 的有效像素拼接成一维张量
        all_probs_cat = torch.cat(all_probs)
        all_targets_cat = torch.cat(all_targets)

        # 1. 按照预测概率从大到小排序
        sorted_probs, indices = torch.sort(all_probs_cat, descending=True)
        sorted_targets = all_targets_cat[indices]

        # 2. 获取正样本 (病灶) 的总像素数
        total_positives = sorted_targets.sum()

        if total_positives > 0:
            # 3. 计算累计真阳性 (Cumulative TP)
            tp = torch.cumsum(sorted_targets, dim=0)

            # 4. 计算每个阈值下的总预测为阳性的数量 (即当前索引号 1, 2, 3...)
            total_preds = torch.arange(1, len(sorted_targets) + 1, device=device, dtype=torch.float32)

            # 5. 计算每个位置的 Precision (TP / (TP + FP))
            precision = tp / total_preds

            # 6. 计算 Average Precision (仅在真实标签为 1 的位置累加 Precision 并取平均)
            # 这等价于黎曼和积分的离散形式，结果与 sklearn 极度接近，且计算速度极快
            aucpr = (torch.sum(precision * sorted_targets) / total_positives).item()
        else:
            aucpr = 0.0

    return confmat, dice.value.item(), aucpr


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, args, print_freq=1, scaler=None):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    # 初始化 RPN 目标生成器
    strides_dict = {'OUT_1': 8, 'OUT_2': 4, 'OUT_3': 2, 'OUT_4': 1}
    rsm_strides = [strides_dict[k] for k in args.PVB_LIST]
    target_gen = TargetGenerator(rsm_strides=rsm_strides, rsm_k=args.rsm_k, pfm_k=args.pfm_k)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)

        # 增加通道维度 [B, 1, H, W]
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # 1. 提取眼球外 Ignore 掩码 (255) 并抹平 target，防止目标生成器爆炸
        ignore_mask = (target == 255.0)
        clean_target = target.clone()
        clean_target[ignore_mask] = 0.0

        # 2. 生成多尺度 Ground Truth
        gt_rsms = target_gen.generate_rsm_batch(clean_target)
        gt_pfm = target_gen.generate_pfm_batch(clean_target)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            # 3. 前向传播
            pred_rsms, pred_pfm = model(image)
            # 4. 计算损失
            loss = criterion(pred_rsms, pred_pfm, gt_rsms, gt_pfm, ignore_mask)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)