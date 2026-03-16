import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import datetime
import json
import argparse
import torch

# 修改 1：引入 RPN 和我们写的 IDRiD 数据集
from src.rpn import RPN
from my_dataset import IDRiDDataset
from train_utils.train_and_eval import train_one_epoch, evaluate, create_lr_scheduler
import transforms as T


# 修改 2：简化 Transform，因为 IDRiDDataset 内部已经处理了 640x640 缩放和 Tensor 转换
class SegmentationPresetTrain:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        trans = []
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    if train:
        return SegmentationPresetTrain(mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)


# 修改 3：创建 RPN 模型
def create_model(args):
    # RPN 为单病灶提取，输出通道数为 1
    model = RPN(
        PVB_LIST=args.PVB_LIST,
        CVB_LIST=args.CVB_LIST,
        n_channels=3,
        n_classes=1,
        bilinear=True
    )
    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    # 二分类，正类通道为1
    num_classes = 1

    # IDRiD 数据集均值
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # 修改 4：使用 IDRiDDataset 替换 DriveDataset
    # 加载数据集，指定为 IDRiDDataset，传入 root_dir 和 split
    train_dataset = IDRiDDataset(
        root_dir=args.data_path,
        split='train',
        lesion_type=args.lesion_type,  # 动态获取配置的病灶类型
        transforms=get_transform(train=True, mean=mean, std=std)
    )

    # 官方的 test 集在此处作为 validation 集使用
    val_dataset = IDRiDDataset(
        root_dir=args.data_path,
        split='test',
        lesion_type=args.lesion_type,
        transforms=get_transform(train=False, mean=mean, std=std)
    )

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(args)
    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    # 修改 5：RPN 论文要求使用 Adam 优化器
    if hasattr(args, 'optimizer') and args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(
            params_to_optimize,
            lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            params_to_optimize,
            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    # lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[250, 350, 400],
        gamma=0.1
    )
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    # 将原本的 best_dice 替换为 best_aucpr
    best_aucpr = 0.
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=None, args=args, print_freq=args.print_freq, scaler=scaler)

        lr_scheduler.step()

        # 【修改 1】：接收 eval 函数返回的 aucpr
        confmat, dice, aucpr = evaluate(model, val_loader, device=device, num_classes=num_classes)

        val_info = str(confmat)
        print(val_info)
        print(f"dice coefficient: {dice:.4f}")
        print(f"AUCPR: {aucpr:.4f}")  # 打印 AUCPR

        # 【修改 2】：将 AUCPR 写入 results 文本日志
        with open(results_file, "a") as f:
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"dice coefficient: {dice:.4f}\n" \
                         f"AUCPR: {aucpr:.4f}\n"
            f.write(train_info + val_info + "\n\n")

        # 【修改 3】：保存最佳模型逻辑更换为 AUCPR
        if args.save_best is True:
            if best_aucpr < aucpr:
                best_aucpr = aucpr
                print(f">> Saved new best model with AUCPR: {best_aucpr:.4f}!")
            else:
                continue  # 如果这一轮没有超越最好成绩，直接跳过保存进入下一轮

            save_file = {"model": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "lr_scheduler": lr_scheduler.state_dict(),
                         "epoch": epoch,
                         "args": args}
            if args.amp:
                save_file["scaler"] = scaler.state_dict()

            if args.save_best is True:
                torch.save(save_file, "save_weights/best_model.pth")
            else:
                torch.save(save_file, "save_weights/model_{}.pth".format(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args(config_file="config.json"):
    # 增加 RPN 特有参数
    default_config = {
        "data_path": "./IDRiD",
        "num_classes": 1,
        "device": "cuda",
        "batch_size": 4,
        "epochs": 500,
        "lr": 0.0001,
        "optimizer": "adam",
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "print_freq": 1,
        "resume": "",
        "start_epoch": 0,
        "save_best": True,
        "amp": False,
        "PVB_LIST": ["OUT_1", "OUT_2"],
        "CVB_LIST": ["OUT_1", "OUT_2", "OUT_3", "OUT_4"],
        "pfm_k": 128,
        "rsm_k": 35
    }

    if os.path.exists(config_file):
        print(f"=> 找到配置文件 '{config_file}'，正在加载...")
        with open(config_file, 'r', encoding='utf-8') as f:
            user_config = json.load(f)
            default_config.update(user_config)
    else:
        print(f"=> 警告: 未找到 '{config_file}'，将完全使用代码内置的默认配置进行训练！")

    args = argparse.Namespace(**default_config)

    print("-" * 50)
    print("Training Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("-" * 50)

    return args


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")
    main(args)