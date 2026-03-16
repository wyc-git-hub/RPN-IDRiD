import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


class IDRiDDataset(Dataset):
    """
    针对 IDRiD 官方数据集结构的自定义 Dataset。
    新增：自动过滤掉没有对应病灶 Mask 的图像对。
    """

    def __init__(self, root_dir, split='train', lesion_type='SE', transforms=None):
        super(IDRiDDataset, self).__init__()
        self.root_dir = root_dir
        self.split = split
        self.lesion_type = lesion_type.upper()
        self.transforms = transforms

        lesion_dict = {
            'MA': '1. Microaneurysms',
            'HE': '2. Haemorrhages',
            'EX': '3. Hard Exudates',
            'SE': '4. Soft Exudates'
        }

        # 路径组合
        if split == 'train':
            img_sub_dir = '1. Original Images/a. Training Set'
            mask_sub_dir = f'2. All Segmentation Groundtruths/a. Training Set/{lesion_dict[self.lesion_type]}'
        else:
            img_sub_dir = '1. Original Images/b. Testing Set'
            mask_sub_dir = f'2. All Segmentation Groundtruths/b. Testing Set/{lesion_dict[self.lesion_type]}'

        self.img_dir = os.path.join(root_dir, img_sub_dir)
        self.mask_dir = os.path.join(root_dir, mask_sub_dir)

        # --- 核心修改：执行预筛选逻辑 ---
        all_imgs = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith('.jpg')])
        self.valid_img_names = []

        print(f"--- 正在筛选包含 {self.lesion_type} 病灶的数据对 ({split}) ---")

        for img_name in all_imgs:
            base_name = os.path.splitext(img_name)[0]
            mask_name = f"{base_name}_{self.lesion_type}.tif"
            mask_path = os.path.join(self.mask_dir, mask_name)

            # 只有当对应的 Mask 文件存在时，才将其加入训练列表
            if os.path.exists(mask_path):
                self.valid_img_names.append(img_name)

        print(f"筛选完成！原始图像数: {len(all_imgs)}, 含有病灶的有效对数: {len(self.valid_img_names)}")

    def _get_foreground_bbox(self, img_pil):
        gray_array = np.array(img_pil.convert("L"))
        mask_fg = gray_array > 10
        rows = np.any(mask_fg, axis=1)
        cols = np.any(mask_fg, axis=0)
        if not np.any(rows) or not np.any(cols):
            return 0, 0, img_pil.width, img_pil.height
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        return xmin, ymin, xmax, ymax

    def __getitem__(self, idx):
        # 使用筛选后的列表
        img_name = self.valid_img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        base_name = os.path.splitext(img_name)[0]
        mask_name = f"{base_name}_{self.lesion_type}.tif"
        mask_path = os.path.join(self.mask_dir, mask_name)

        img = Image.open(img_path).convert("RGB")
        bbox = self._get_foreground_bbox(img)
        img = img.crop(bbox)

        # 既然已经筛选过，这里直接读取即可
        mask = Image.open(mask_path).convert("RGB")
        mask = mask.crop(bbox)

        # 缩放处理
        target_size = (640, 640)
        img = img.resize(target_size, resample=Image.BILINEAR)
        mask = mask.resize(target_size, resample=Image.NEAREST)

        # 提取黑边 (255)
        img_gray_resized = np.array(img.convert("L"))
        out_of_eyeball_mask = img_gray_resized <= 10

        # 二值化
        mask_array = np.array(mask)
        mask_gray = np.max(mask_array, axis=2)

        mask_label = np.zeros_like(mask_gray, dtype=np.float32)
        mask_label[mask_gray > 0] = 1.0
        mask_label[out_of_eyeball_mask] = 255.0

        img_tensor = TF.to_tensor(img)
        mask_tensor = torch.from_numpy(mask_label).unsqueeze(0)

        if self.transforms is not None:
            img_tensor, mask_tensor = self.transforms(img_tensor, mask_tensor)

        return img_tensor, mask_tensor

    def __len__(self):
        return len(self.valid_img_names)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets