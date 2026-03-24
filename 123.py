# ====================== 安装依赖 ======================
# pip install torch torchvision torchaudio timm librosa numpy soundfile scikit-learn

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
import soundfile as sf
from tqdm import tqdm
import random


# ====================== 1. 特征提取（3通道MFCC） ======================
def extract_mfcc_3ch(audio_path, sr=22050, n_mfcc=40, max_len=128):
    y, sr = sf.read(audio_path)
    if len(y.shape) > 1:  # 多通道转单声道
        y = np.mean(y, axis=1)

    # 固定长度（填充或截断）
    if len(y) > sr * 4:  # 最长4秒
        y = y[:sr * 4]
    else:
        y = np.pad(y, (0, sr * 4 - len(y)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=512)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # 转成图像格式 (3, H, W)
    image = np.stack([mfcc, delta, delta2], axis=0)

    # 时间维度统一到 max_len
    if image.shape[2] > max_len:
        image = image[:, :, :max_len]
    else:
        pad_width = max_len - image.shape[2]
        image = np.pad(image, ((0, 0), (0, 0), (0, pad_width)))

    # 归一化到 [0,1]
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return torch.from_numpy(image).float()  # shape: (3, 40, 128)


# ====================== 2. SpecAugment（谱图专用增强） ======================
class SpecAugment(nn.Module):
    def __init__(self, freq_mask=8, time_mask=20, p=0.8):
        super().__init__()
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.p = p

    def forward(self, x):  # x: (B, 3, H, W)
        if random.random() > self.p:
            return x
        B, C, H, W = x.shape
        # 频率掩码
        for _ in range(random.randint(1, 2)):
            f = random.randint(0, self.freq_mask)
            f0 = random.randint(0, H - f)
            x[:, :, f0:f0 + f, :] = 0
        # 时间掩码
        for _ in range(random.randint(1, 2)):
            t = random.randint(0, self.time_mask)
            t0 = random.randint(0, W - t)
            x[:, :, :, t0:t0 + t] = 0
        return x


# ====================== 3. Dataset ======================
class AudioDataset(Dataset):
    def __init__(self, file_list, label_list, transform=None):
        self.file_list = file_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = extract_mfcc_3ch(self.file_list[idx])  # (3, 40, 128)
        img = img.unsqueeze(0)  # 临时加batch维给transform
        if self.transform:
            img = self.transform(img)
        img = img.squeeze(0)
        return img, self.label_list[idx]


# ====================== 4. 模型（MobileViT_xxs） ======================
def get_model(num_classes=10):
    model = timm.create_model('mobilevit_xxs', pretrained=True, num_classes=num_classes)

    # MobileViT默认输入3通道，已完美匹配我们的3通道MFCC
    # 如果想改成单通道Log-Mel，只需把下面一行取消注释并改特征提取
    # model.stem.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)

    return model


# ====================== 5. 训练主函数 ======================
def train_model(data_dir, num_classes=10, epochs=30, batch_size=32, lr=1e-4):
    # 扫描数据集（文件夹结构：data/class0/xxx.wav）
    file_list, label_list = [], []
    class_to_idx = {}
    for idx, class_name in enumerate(sorted(os.listdir(data_dir))):
        class_to_idx[class_name] = idx
        class_path = os.path.join(data_dir, class_name)
        for fname in os.listdir(class_path):
            if fname.endswith(('.wav', '.WAV')):
                file_list.append(os.path.join(class_path, fname))
                label_list.append(idx)

    # 划分数据集
    train_files, val_files, train_labels, val_labels = train_test_split(
        file_list, label_list, test_size=0.2, stratify=label_list, random_state=42)

    # 增强（只在训练时用）
    train_transform = T.Compose([
        SpecAugment(freq_mask=8, time_mask=25, p=0.85),
        T.RandomErasing(p=0.5, scale=(0.02, 0.1)),  # 额外随机擦除
    ])

    train_dataset = AudioDataset(train_files, train_labels, transform=train_transform)
    val_dataset = AudioDataset(val_files, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = get_model(num_classes).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

        # 验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.cuda(), labels.cuda()
                outputs = model(imgs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Epoch {epoch + 1} Val Acc: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_mobilevit_mfcc.pth")
        scheduler.step()

    print(f"训练完成！最佳验证准确率: {best_acc:.2f}%")
    return model


# ====================== 启动训练 ======================
if __name__ == "__main__":
    DATA_DIR = "你的数据集路径"  # 示例: "data/UrbanSound8K"
    model = train_model(DATA_DIR, num_classes=10, epochs=30, batch_size=32)