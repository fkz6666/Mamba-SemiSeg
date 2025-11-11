import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from zdata_loader import GeoTIFFDataset
from crf_postprocess import apply_crf
from mamba_ssm.modules.mamba_simple import Mamba
import numpy as np
from PIL import Image
import torch.nn as nn
import cv2

# ======================
# 模型定义
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
                patch = x4[:, :, i:i+patch_size, j:j+patch_size]
                if patch.shape[-1] < patch_size or patch.shape[-2] < patch_size:
                    continue
                flat = patch.reshape(B, 128, -1).permute(0, 2, 1)
                out = self.mamba(flat)
                out = out.permute(0, 2, 1).reshape(B, 128, patch_size, patch_size)
                patch_out[:, :, i:i+patch_size, j:j+patch_size] = out

        m_out = patch_out
        up1 = self.up1(m_out)
        cat1 = torch.cat([up1, x3], dim=1)
        dec1 = self.dec1(cat1)
        up2 = self.up2(dec1)
        cat2 = torch.cat([up2, x1], dim=1)
        out = self.dec2(cat2)

        # ✅ 输出概率轻微平滑
        out = torch.nn.functional.avg_pool2d(out, kernel_size=3, stride=1, padding=1)

        return out

# ======================
# 配置
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thresh = 0.6

# 加载模型
model = MambaSegmentation().to(device)
model.load_state_dict(torch.load("checkpoints/initial001.pth", map_location=device))
model.eval()

# 数据加载
dataset = GeoTIFFDataset(label_mode=False)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# 输出路径
output_dir = r"C:\Users\10759\Desktop\6990001_ps"
os.makedirs(output_dir, exist_ok=True)

# ======================
# 伪标签生成 + CRF
# ======================
@torch.no_grad()
def main():
    for img, path in loader:
        img = img.to(device)
        output = model(img)  # [1,2,H,W]
        prob = F.softmax(output, dim=1)[0].cpu().numpy()  # [2,H,W]

        image_np = img[0].cpu().numpy().transpose(1, 2, 0) * 255
        image_np = image_np.astype(np.uint8).copy()

        # CRF 后处理
        refined = apply_crf(image_np, prob)
        H, W = prob.shape[1], prob.shape[2]
        refined = refined.reshape(H, W)

        # 置信度过滤
        conf_mask = (prob[1] > thresh).astype(np.uint8)
        final_mask = (refined * conf_mask) * 255

        # ✅ 可选：闭运算去除小黑洞 / 毛刺
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE,
                                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                                      iterations=1)

        filename = os.path.basename(path[0]).replace(".tif", ".png")
        save_path = os.path.join(output_dir, filename)
        Image.fromarray(final_mask.astype(np.uint8)).save(save_path)

        print(f"✅ 生成伪标签: {save_path}")

if __name__ == "__main__":
    main()


