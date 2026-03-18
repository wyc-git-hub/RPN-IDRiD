import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

# 导入你自己的模块 (请确保路径和名称与你的项目一致)
from my_dataset import IDRiDDataset
import transforms as T
from src.rpn import RPN
import train_utils.distributed_utils as utils


def get_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    # 测试时不需要几何数据增强(翻转/旋转等)，但【必须保留直方图均衡化预处理】
    transforms = [
        T.MaskedHistogramEqualization(),  # 核心预处理：增强对比度，凸显病灶
        # 如果你的 my_dataset.py 没有做 ToTensor 操作，这里需要加上 T.ToTensor()
        T.Normalize(mean=mean, std=std)
    ]
    return T.Compose(transforms)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 创建保存可视化结果的文件夹
    save_dir = os.path.join("test_results", args.lesion_type)
    os.makedirs(save_dir, exist_ok=True)

    # 2. 加载测试数据集
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    test_dataset = IDRiDDataset(
        root_dir=args.data_path,
        split='test',
        lesion_type=args.lesion_type,
        transforms=get_transform(mean=mean, std=std)
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=args.num_workers, collate_fn=test_dataset.collate_fn)

    # 3. 初始化模型并加载权重
    model = RPN(n_channels=3, n_classes=1,
                PVB_LIST=args.PVB_LIST, CVB_LIST=args.CVB_LIST)

    assert os.path.exists(args.weights), f"Weights file {args.weights} not found!"
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    # 切换至评估模式
    model.eval()

    # 4. 初始化评估指标记录器
    confmat = utils.ConfusionMatrix(2)
    dice_metric = utils.DiceCoefficient(num_classes=2, ignore_index=255)

    all_probs = []
    all_targets = []

    print(f"--- 开始测试 {args.lesion_type} 病灶，共 {len(test_dataset)} 张图像 ---")

    # 使用 inference_mode 代替 no_grad，推理速度更快，显存占用更低
    with torch.inference_mode():
        for idx, (image, target) in enumerate(tqdm(test_loader, desc="Testing")):
            image = image.to(device)
            target = target.to(device)

            target_long = target.long()
            if target_long.dim() == 4:
                target_long = target_long.squeeze(1)

            # --- 前向传播 ---
            # 获取 PVB (RSM) 和 CVB (PFM) 的预测
            pred_rsms, pred_pfm = model(image)

            # --- 概率转换与对齐 ---
            # 1. 获取 CVB 细节概率 (网络输出是 logits，需要 Sigmoid)
            cvb_prob = torch.sigmoid(pred_pfm).squeeze(1)

            # 2. 获取 PVB 大局概率 (训练时用的 BCE 而非 BCEWithLogits，说明内部已过 Sigmoid)
            rsm_prob = pred_rsms[0].squeeze(1)

            # 3. 尺寸对齐 (使用双线性插值将 PVB 的小尺寸输出放大至原图大小)
            if rsm_prob.shape != cvb_prob.shape:
                rsm_prob_up = F.interpolate(
                    rsm_prob.unsqueeze(1),
                    size=cvb_prob.shape[-2:],
                    mode='bilinear',
                    align_corners=True
                ).squeeze(1)
            else:
                rsm_prob_up = rsm_prob

            # --- 核心修复：PVB 掩码抑制 CVB 背景噪声 ---
            # 设定周边视觉容忍度：只要 PVB 觉得有一点点像病灶(>0.15)，就允许 CVB 去发挥
            rsm_mask = (rsm_prob_up > 0.15).float()
            # 掩码乘法(连续域逻辑与)：掩码外归 0 杀灭噪声，掩码内保留 100% CVB 置信度
            output_prob = cvb_prob * rsm_mask

            # --- 指标收集 ---
            valid_mask = target_long != 255
            all_probs.append(output_prob[valid_mask])
            all_targets.append(target_long[valid_mask].float())

            # 阈值切分
            pred_label = (output_prob > 0.5).long()

            # 更新指标
            pred_for_dice = torch.stack([1 - output_prob, output_prob], dim=1)
            confmat.update(target_long.flatten(), pred_label.flatten())
            dice_metric.update(pred_for_dice, target_long)

            # --- 可视化保存 ---
            img_name = test_dataset.valid_img_names[idx]
            base_name = os.path.splitext(img_name)[0]

            # 1. 保存最终融合后的细节概率图 (不再花白)
            prob_map = (output_prob[0].cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(prob_map).save(os.path.join(save_dir, f"{base_name}_prob.png"))

            # 2. 保存最终二值化预测图 (阈值 0.5)
            pred_mask_img = (pred_label[0].cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(pred_mask_img).save(os.path.join(save_dir, f"{base_name}_pred.png"))

            # 3. 保存真实的 Ground Truth 图
            gt_array = target_long[0].cpu().numpy()
            gt_vis = np.zeros_like(gt_array, dtype=np.uint8)
            gt_vis[gt_array == 1] = 255
            gt_vis[gt_array == 255] = 128  # 灰色表示眼球外的忽略区域
            Image.fromarray(gt_vis).save(os.path.join(save_dir, f"{base_name}_gt.png"))

            # 4. 【新增】保存 PVB 的周边视觉光晕图 (展示大体病灶区域的定位能力)
            rsm_map = (rsm_prob_up[0].cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(rsm_map).save(os.path.join(save_dir, f"{base_name}_rsm_prob.png"))

    # 5. 汇总并计算最终指标
    confmat.reduce_from_all_processes()
    dice_metric.reduce_from_all_processes()

    # 纯 PyTorch AUCPR 计算
    all_probs_cat = torch.cat(all_probs)
    all_targets_cat = torch.cat(all_targets)
    sorted_probs, indices = torch.sort(all_probs_cat, descending=True)
    sorted_targets = all_targets_cat[indices]
    total_positives = sorted_targets.sum()

    if total_positives > 0:
        tp = torch.cumsum(sorted_targets, dim=0)
        total_preds = torch.arange(1, len(sorted_targets) + 1, device=device, dtype=torch.float32)
        precision = tp / total_preds
        aucpr = (torch.sum(precision * sorted_targets) / total_positives).item()
    else:
        aucpr = 0.0

    # 6. 打印最终结果
    print("\n" + "=" * 40)
    print(f"  Test Results for {args.lesion_type}  ")
    print("=" * 40)
    print(confmat)
    print(f"Dice Coefficient: {dice_metric.value.item():.4f}")
    print(f"AUCPR:            {aucpr:.4f}")
    print("=" * 40)
    print(f"可视化结果已保存至: {save_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Test RPN Model on IDRiD")
    parser.add_argument("--data-path", default="./data/IDRiD/A. Segmentation", help="IDRiD dataset root path")
    parser.add_argument("--lesion-type", default="SE", type=str, help="lesion type: MA, HE, EX, SE")
    parser.add_argument("--weights", default="./save_weights/best_model.pth", help="path to saved best weights")
    parser.add_argument("--num-workers", default=4, type=int, help="number of data loading workers")
    parser.add_argument("--device", default="cuda", help="device (cuda or cpu)")

    parser.add_argument("--PVB_LIST", default=["OUT_3", "OUT_2"], nargs='+', help="PVB output layers")
    parser.add_argument("--CVB_LIST", default=["OUT_3", "OUT_2"], nargs='+', help="CVB layers")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)