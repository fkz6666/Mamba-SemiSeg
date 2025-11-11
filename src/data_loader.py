import os
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
from PIL import Image
import cv2  # 新增，用于高斯平滑

class GeoTIFFDataset(Dataset):
    def __init__(self, label_mode=True, pseudo_path=None):
        """
        数据目录固定为：
        C:/Users/10759/Desktop/mamba_semi_supervised_internal_wave/dataset/labeled
        label_mode=True 表示加载标签（训练）
        pseudo_path：指定伪标签目录
        """
        self.root_dir = r"C:/Users/10759/Desktop/内波项目2/mamba_semi_supervised_internal_wave/dataset/labeled"
        #self.root_dir = r"C:\Users\10759\Desktop\z图片tif3"
        self.label_mode = label_mode
        self.pseudo_path = pseudo_path
        self.file_list = [f for f in os.listdir(self.root_dir) if f.endswith('.tif')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        name = self.file_list[idx]
        image_path = os.path.join(self.root_dir, name)

        # ✅ 读取遥感图像并归一化到 0~1
        with rasterio.open(image_path) as src:
            img = src.read(out_dtype=np.float32) / 255.0   # [C,H,W]
        img = torch.tensor(img, dtype=torch.float32)

        # 如果是单通道，重复成 3 通道
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        # 推理模式（不加载标签）
        if not self.label_mode:
            return img, image_path

        # ✅ 标签读取（优先伪标签路径）
        label_dir = self.pseudo_path if self.pseudo_path else self.root_dir
        label_path = os.path.join(label_dir, name.replace('.tif', '.png'))
        label_np = np.array(Image.open(label_path).convert('L'), dtype=np.uint8)

        # ✅ 标签高斯平滑（仅训练模式），sigma=1 较轻微
        label_np = cv2.GaussianBlur(label_np, (3, 3), sigmaX=1)

        # 转为 Long 类型张量
        label = torch.tensor(label_np, dtype=torch.long)

        return img, label

