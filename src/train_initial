# ✅ train_initial.py：轻量级 patch-wise MambaSegmentation 初始训练（带权重、进度条、耗时、绘图）
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from zdata_loader import GeoTIFFDataset
from mamba_ssm.modules.mamba_simple import Mamba

# ======================
# 边界损失函数（Boundary Loss）
# ======================
def boundary_loss(pred, target):
    pred_probs = torch.softmax(pred, dim=1)[:, 1, :, :]  # 前景通道
    target_bin = (target > 0).float()

    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32, device=pred.device).unsqueeze(0).unsqueeze(0)
    sobel_y = sobel_x.transpose(2, 3)

    pred_grad_x = torch.nn.functional.conv2d(pred_probs.unsqueeze(1), sobel_x, padding=1)
    pred_grad_y = torch.nn.functional.conv2d(pred_probs.unsqueeze(1), sobel_y, padding=1)
    target_grad_x = torch.nn.functional.conv2d(target_bin.unsqueeze(1), sobel_x, padding=1)
    target_grad_y = torch.nn.functional.conv2d(target_bin.unsqueeze(1), sobel_y, padding=1)

    pred_grad = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-6)
    target_grad = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-6)

    return torch.mean((pred_grad - target_grad) ** 2)

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
# 训练配置
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = GeoTIFFDataset(label_mode=True)
loader = DataLoader(dataset, batch_size=4, shuffle=True)
model = MambaSegmentation().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 类别权重
weight = torch.tensor([0.2, 2.0], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weight)

os.makedirs("checkpoints", exist_ok=True)

# ======================
# 训练主循环
# ======================
num_epochs = 20
best_loss = float("inf")
loss_history = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    start_time = time.time()

    pbar = tqdm(loader, desc=f"[Epoch {epoch+1}/{num_epochs}]", leave=False)
    for img, mask in pbar:
        img, mask = img.to(device), mask.to(device)
        out = model(img)

        # 主损失 + 边界损失
        loss_main = criterion(out, mask)
        loss_b = boundary_loss(out, mask)
        loss = loss_main + 0.1 * loss_b  # 0.1 可调

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(loader)
    loss_history.append(avg_loss)
    elapsed = time.time() - start_time
    print(f"✅ Epoch {epoch+1:02d}/{num_epochs} - Loss: {avg_loss:.4f} - Time: {elapsed:.2f}s")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "checkpoints/best_initial001.pth")
        print(f"🎯 Best model updated at Epoch {epoch+1}")

# 保存最终模型
torch.save(model.state_dict(), "checkpoints/initial001.pth")
print("✅ 初始训练完成，已保存 final 模型到 checkpoints/initial001.pth")

# 绘制 loss 曲线
plt.plot(loss_history, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.legend()
plt.savefig("checkpoints/loss_curve001.png")
print("📉 Loss 曲线图已保存至 checkpoints/loss_curve001.png")



