import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from zdata_loader import GeoTIFFDataset
from mamba_ssm.modules.mamba_simple import Mamba
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import cv2

# ======================
# 模型定义（增加概率平滑）
# ======================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class MambaSegmentation(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.enc1 = ConvBlock(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.mamba = Mamba(d_model=128, d_state=16, expand=2, bias=False)
        self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128 + 128, 64)
        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Conv2d(64 + 64, num_classes, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x1 = self.enc1(x)
        x2 = self.pool1(x1)
        x3 = self.enc2(x2)
        x4 = self.pool2(x3)

        patch_out = torch.zeros_like(x4)
        patch_size = 16
        for i in range(0, x4.shape[2], patch_size):
            for j in range(0, x4.shape[3], patch_size):
                patch = x4[:, :, i:i + patch_size, j:j + patch_size]
                if patch.shape[-1] < patch_size or patch.shape[-2] < patch_size:
                    continue
                flat = patch.reshape(B, 128, -1).permute(0, 2, 1)
                out = self.mamba(flat)
                out = out.permute(0, 2, 1).reshape(B, 128, patch_size, patch_size)
                patch_out[:, :, i:i + patch_size, j:j + patch_size] = out

        m_out = patch_out
        up1 = self.up1(m_out)
        cat1 = torch.cat([up1, x3], dim=1)
        dec1 = self.dec1(cat1)
        up2 = self.up2(dec1)
        cat2 = torch.cat([up2, x1], dim=1)
        out = self.dec2(cat2)

        # ✅ 概率轻微平滑（保持与训练一致）
        out = torch.nn.functional.avg_pool2d(out, kernel_size=3, stride=1, padding=1)

        return out

# ======================
# 推理配置
# ======================
infer_output_dir = r"C:\Users\10759\Desktop\6990001_lv"
os.makedirs(infer_output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = GeoTIFFDataset(label_mode=False)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# 加载模型
model = MambaSegmentation().to(device)
model_path = r"D:\my_python\checkpoints\finetuned001.69_9.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ======================
# 滤波增强函数（保持原有逻辑）
# ======================
def refine_prediction(mask):
    mask = (mask > 0).astype(np.uint8) * 255

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=2)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned)
    img_area = mask.shape[0] * mask.shape[1]
    min_area = max(30, img_area * 0.0005)
    max_allow_area = img_area * 0.03

    filtered = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(w, h) / (min(w, h) + 1e-5)
        solidity = area / (w * h + 1e-5)

        is_stripe = (aspect > 2.5) or (aspect > 1.8 and solidity < 0.6)
        if area >= min_area and (is_stripe or area > 800):
            component = (labels == i).astype(np.uint8) * 255
            if aspect > 3.0:
                if w > h:
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
                else:
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
                component = cv2.dilate(component, kernel, iterations=1)
                component = cv2.morphologyEx(component, cv2.MORPH_CLOSE, kernel, iterations=1)
            else:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                component = cv2.morphologyEx(component, cv2.MORPH_CLOSE, kernel, iterations=2)
            filtered = cv2.bitwise_or(filtered, component)

    filtered_bin = (filtered > 0).astype(np.uint8)
    contours, _ = cv2.findContours(filtered_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            cv2.drawContours(filtered_bin, [cnt], 0, 1, -1)

    filtered_bin = cv2.medianBlur(filtered_bin, 3)

    distance = cv2.distanceTransform(255 - (filtered_bin * 255), cv2.DIST_L2, 3)
    max_distance = np.max(distance)
    if max_distance > 5:
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        _, distance_thresh = cv2.threshold(distance, 0.5 * max_distance, 255, cv2.THRESH_BINARY)
        distance_thresh = distance_thresh.astype(np.uint8)
        connected = cv2.morphologyEx(filtered_bin, cv2.MORPH_CLOSE, kernel_connect, iterations=2)
        filtered_bin = cv2.bitwise_and(connected, (distance_thresh > 0).astype(np.uint8)) | filtered_bin

    smoothed = cv2.GaussianBlur(filtered_bin * 255, (5, 5), sigmaX=1.5)
    _, final = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)

    kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final = cv2.morphologyEx(final, cv2.MORPH_OPEN, kernel_final, iterations=1)
    final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kernel_final, iterations=1)

    return final

# ======================
# 推理主函数
# ======================
@torch.no_grad()
def infer():
    for img, path in tqdm(loader, desc="[Infering Images]"):
        img = img.to(device)
        out = model(img)

        probs = F.softmax(out, dim=1)
        pred = torch.argmax(probs, dim=1)[0].cpu().numpy()

        filtered = refine_prediction(pred * 255)

        filename = os.path.basename(path[0]).replace(".tif", ".png")
        save_path = os.path.join(infer_output_dir, filename)
        Image.fromarray(filtered).save(save_path)
        print(f"✅ 已保存结果: {save_path}")

if __name__ == "__main__":
    debug_dir = os.path.join(infer_output_dir, "debug3")
    os.makedirs(debug_dir, exist_ok=True)
    infer_output_dir = debug_dir
    infer()


