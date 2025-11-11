import os
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import segmentation
from skimage.metrics import adapted_rand_error  # 用于边界一致性评估
from skimage.measure import label

# ✅ 路径设置
pred_dir = r"C:\Users\10759\Desktop\黑白"
label_dir = r"C:\Users\10759\Desktop\183对照"
image_names = [f for f in os.listdir(pred_dir) if f.endswith('.png')]

# ✅ 掩码读取函数
def load_mask(path, is_label=False):
    img = Image.open(path)
    img = img.convert('L')  # 灰度
    mask = np.array(img)
    if is_label:
        return (mask > 30).astype(np.uint8)
    else:
        return (mask > 127).astype(np.uint8)

# ✅ 边界F1-score计算函数（常见近似方式）
def boundary_f1_score(pred, label):
    pred_bound = segmentation.find_boundaries(pred, mode='thick').astype(np.uint8)
    label_bound = segmentation.find_boundaries(label, mode='thick').astype(np.uint8)

    TP = np.sum((pred_bound == 1) & (label_bound == 1))
    FP = np.sum((pred_bound == 1) & (label_bound == 0))
    FN = np.sum((pred_bound == 0) & (label_bound == 1))

    eps = 1e-6
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    bf1 = 2 * precision * recall / (precision + recall + eps)

    return bf1

# ✅ 可视化函数
def visualize_sample(pred, label, name, pixel_acc, iou, dice, precision, recall, bf1, cm):
    plt.figure(figsize=(18, 5))

    plt.subplot(131)
    plt.imshow(pred, cmap='gray')
    plt.title('Prediction')

    plt.subplot(132)
    plt.imshow(label, cmap='gray')
    plt.title('Label')

    overlay = np.zeros((*label.shape, 3), dtype=np.uint8)
    overlay[(label == 1) & (pred == 1)] = [0, 0, 255]      # TP - 蓝色
    overlay[(label == 1) & (pred == 0)] = [255, 0, 0]      # FN - 红色
    overlay[(label == 0) & (pred == 1)] = [0, 255, 255]    # FP - 青色

    plt.subplot(133)
    plt.imshow(overlay)
    plt.title('Red=FN  Blue=TP  Cyan=FP')

    plt.suptitle(
        f"Sample: {name} | PixelAcc: {pixel_acc:.4f}  IoU: {iou:.4f}  Dice: {dice:.4f}\n"
        f"Precision: {precision:.4f}  Recall: {recall:.4f}  BF1: {bf1:.4f}\n"
        f"Confusion Matrix:\n[[{cm[0,0]} {cm[0,1]}]\n [{cm[1,0]} {cm[1,1]}]]",
        fontsize=10
    )

    plt.tight_layout()
    plt.show()

# ✅ 全量评估
for name in tqdm(image_names, desc="Evaluating all samples"):
    pred_path = os.path.join(pred_dir, name)
    label_path = os.path.join(label_dir, name)

    if not os.path.exists(label_path):
        print(f"❌ 缺少标注文件: {label_path}")
        continue

    pred = load_mask(pred_path, is_label=False)
    label = load_mask(label_path, is_label=True)

    # 计算每张图像的混淆矩阵与指标
    cm = confusion_matrix(label.flatten(), pred.flatten(), labels=[0, 1])
    TP = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TN = cm[0, 0]

    eps = 1e-6
    pixel_acc = (TP + TN) / (TP + FP + FN + TN + eps)
    iou = TP / (TP + FP + FN + eps)
    dice = 2 * TP / (2 * TP + FP + FN + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    bf1 = boundary_f1_score(pred, label)

    # 输出数值
    print(f"\n📄 Sample: {name}")
    print(f"Pixel Accuracy: {pixel_acc:.4f}")
    print(f"IoU: {iou:.4f}")
    print(f"Dice: {dice:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Boundary F1 (BF1): {bf1:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    # 可视化
    visualize_sample(pred, label, name, pixel_acc, iou, dice, precision, recall, bf1, cm)

