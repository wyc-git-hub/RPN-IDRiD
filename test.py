import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import json
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

# 导入你自己的模块 (请确保路径和名称与你的项目一致)
from my_dataset import IDRiDDataset
import transforms as T
from src.rpn import RPN  # 假设你的模型类在这个文件里，请根据实际情况修改
import train_utils.distributed_utils as utils


def get_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    # 测试时只需要转 Tensor 和标准化，不需要数据增强
    transforms = [T.Normalize(mean=mean, std=std)]
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
    # 请根据你的实际网络定义接口修改这里的参数
    model = RPN(n_channels=3, n_classes=1,
                      PVB_LIST=args.PVB_LIST, CVB_LIST=args.CVB_LIST)

    # 加载 best_model 权重
    assert os.path.exists(args.weights), f"Weights file {args.weights} not found!"
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    # 4. 初始化评估指标记录器
    confmat = utils.ConfusionMatrix(2)
    dice_metric = utils.DiceCoefficient(num_classes=2, ignore_index=255)

    all_probs = []
    all_targets = []

    print(f"--- 开始测试 {args.lesion_type} 病灶，共 {len(test_dataset)} 张图像 ---")

    with torch.no_grad():
        for idx, (image, target) in enumerate(tqdm(test_loader, desc="Testing")):
            image = image.to(device)
            target = target.to(device)

            target_long = target.long()
            if target_long.dim() == 4:
                target_long = target_long.squeeze(1)

            # 前向传播 (我们只关心最终的 PFM 预测)
            _, pred_pfm = model(image)

            # 将 logits 转换为 0~1 的概率
            output_prob = torch.sigmoid(pred_pfm).squeeze(1)

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
            # 获取原图的文件名 (从 dataset 中获取)
            img_name = test_dataset.valid_img_names[idx]
            base_name = os.path.splitext(img_name)[0]

            # 1. 保存概率图 (可以看清模型是在哪里犹豫)
            prob_map = (output_prob[0].cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(prob_map).save(os.path.join(save_dir, f"{base_name}_prob.png"))

            # 2. 保存二值化预测图 (阈值 0.5)
            pred_mask = (pred_label[0].cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(pred_mask).save(os.path.join(save_dir, f"{base_name}_pred.png"))

            # 3. 保存真实的 Ground Truth 图 (方便对比)
            # 把 255 (忽略区) 变成灰色，1 (病灶) 变成纯白，0 (背景) 变成纯黑
            gt_array = target_long[0].cpu().numpy()
            gt_vis = np.zeros_like(gt_array, dtype=np.uint8)
            gt_vis[gt_array == 1] = 255
            gt_vis[gt_array == 255] = 128  # 灰色表示眼球外的忽略区域
            Image.fromarray(gt_vis).save(os.path.join(save_dir, f"{base_name}_gt.png"))

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

    # 请确保这里的结构与你训练时保持一致
    parser.add_argument("--PVB_LIST", default=["OUT_3", "OUT_2"], nargs='+', help="PVB output layers")
    parser.add_argument("--CVB_LIST", default=["OUT_3", "OUT_2"], nargs='+', help="CVB layers")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)