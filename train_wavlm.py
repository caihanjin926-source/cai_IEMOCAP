import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# 导入新的 Dataset 和 原来的 Model
from dataset_wavlm import WavLMDataset, pad_collate_wavlm
from model import AudioMobileViT

# ================= 全局超参数 =================
BATCH_SIZE = 16
LEARNING_RATE = 5e-5  # 使用了更高级的特征，学习率可以稍微调小一点，防止步子迈太大
EPOCHS = 20
CSV_PATH = "D:\mobile_audio\DataSet\IEMOCAP_Labels\my_iemocap_labels_6class.csv"
FEATURE_DIR = r"D:\mobile_audio\DataSet\wavlm_features"  # 指向你刚提取的特征文件夹
SAVE_PATH = "best_mobilevit_wavlm.pth"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 当前使用的计算设备: {device}")

    # ================= 加载数据 =================
    print("正在加载 WavLM 特征数据集...")
    full_dataset = WavLMDataset(csv_file=CSV_PATH, feature_dir=FEATURE_DIR)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 注意这里使用了新的 pad_collate_wavlm
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate_wavlm)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate_wavlm)

    print(f"训练集: {len(train_dataset)} | 验证集: {len(val_dataset)}")

    # ================= 初始化模型 =================
    model = AudioMobileViT(num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    best_val_acc = 0.0

    # ================= 开始训练 =================
    for epoch in range(EPOCHS):
        print(f"\n[{epoch + 1}/{EPOCHS}] 开始训练...")
        model.train()
        running_loss = 0.0

        train_bar = tqdm(train_loader, desc="Training")
        for features, labels in train_bar:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)

        # ================= 验证阶段 =================
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="Validating")
            for features, labels in val_bar:
                features, labels = features.to(device), labels.to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total

        print(
            f"Epoch {epoch + 1} 总结 -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"⭐ 发现更好的模型！权重已保存至 {SAVE_PATH}")

    print(f"\n🎉 WavLM + MobileViT 训练完成！最高验证集准确率: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()