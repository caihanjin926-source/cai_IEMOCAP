import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
from torchaudio.transforms import FrequencyMasking, TimeMasking

# 导入我们之前写好的组件 (确保 dataset.py 和 model.py 在同级目录下)
from dataset import IEMOCAPDataset, pad_collate
from model import AudioMobileViT

# ================= 1. 全局超参数设置 =================
BATCH_SIZE = 16  # 每次喂给模型的音频数量
LEARNING_RATE = 1e-4  # 学习率 (Transformer 架构通常用小一点的学习率)
EPOCHS = 10  # 训练轮数
CSV_PATH = "D:\mobile_audio\DataSet\IEMOCAP_Labels\my_iemocap_labels_6class.csv"
IEMOCAP_ROOT = r"D:\mobile_audio\DataSet\IEMOCAP_full_release"
SAVE_PATH = "best_mobilevit_audio.pth"  # 最优模型权重保存路径


def main():
    # ================= 2. 准备设备 =================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 当前使用的计算设备: {device}")

    # 新增：初始化 SpecAugment 数据增强 (放在 device 下面即可)
    # 随机遮挡最多 10 个频率通道，最多遮挡 30 个时间帧
    freq_mask = FrequencyMasking(freq_mask_param=10).to(device)
    time_mask = TimeMasking(time_mask_param=30).to(device)

    # ================= 3. 加载数据 =================
    print("正在加载数据集...")
    full_dataset = IEMOCAPDataset(csv_file=CSV_PATH, iemocap_root=IEMOCAP_ROOT)

    # 80% 训练，20% 验证
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate)

    print(f"训练集: {len(train_dataset)} 个样本 | 验证集: {len(val_dataset)} 个样本")

    # ================= 4. 初始化模型、损失函数和优化器 =================
    model = AudioMobileViT(num_classes=6).to(device)

    # 交叉熵损失 (适合多分类任务)
    criterion = nn.CrossEntropyLoss()
    # AdamW 优化器 (对 Vision Transformer 系列非常友好)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    best_val_acc = 0.0  # 记录最高准确率

    # ================= 5. 开始训练循环 =================
    for epoch in range(EPOCHS):
        print(f"\n[{epoch + 1}/{EPOCHS}] 开始训练...")
        model.train()  # 设置为训练模式
        running_loss = 0.0

        # 训练阶段
        train_bar = tqdm(train_loader, desc="Training")
        for mels, labels in train_bar:
            mels, labels = mels.to(device), labels.to(device)

            # 新增：仅在训练时应用 SpecAugment (随机擦除一部分特征)
            mels = freq_mask(mels)
            mels = time_mask(mels)

            optimizer.zero_grad()  # 梯度清零
            outputs = model(mels)  # 前向传播
            loss = criterion(outputs, labels)  # 计算 Loss
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            running_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)

        # ================= 6. 验证阶段 =================
        model.eval()  # 设置为评估模式 (关闭 Dropout 等)
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  # 验证阶段不需要计算梯度，省显存
            val_bar = tqdm(val_loader, desc="Validating")
            for mels, labels in val_bar:
                mels, labels = mels.to(device), labels.to(device)

                outputs = model(mels)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # 获取预测类别 (概率最大的那个)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total

        print(
            f"Epoch {epoch + 1} 总结 -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # 保存表现最好的模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"⭐ 发现更好的模型！权重已保存至 {SAVE_PATH}")

    print("\n🎉 训练全部完成！最高验证集准确率: {:.2f}%".format(best_val_acc))


if __name__ == "__main__":
    main()
