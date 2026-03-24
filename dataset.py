import os
import pandas as pd
import torch
import torchaudio
import soundfile as sf  # 新增这一行
from torch.utils.data import Dataset, DataLoader


class IEMOCAPDataset(Dataset):
    def __init__(self, csv_file, iemocap_root, transform=None):
        """
        csv_file: 你刚刚生成的 my_iemocap_labels_6class.csv 的路径
        iemocap_root: IEMOCAP 数据集的根目录路径
        """
        self.data_frame = pd.read_csv(csv_file)
        self.iemocap_root = iemocap_root
        self.transform = transform

        # 将文本标签映射为数字 (0-5)
        self.emotion_to_idx = {
            'ang': 0, 'exc': 1, 'fru': 2,
            'hap': 3, 'neu': 4, 'sad': 5
        }

        # 音频特征提取器 (提取 Mel 频谱)
        # MobileViT 喜欢 2D 的图像输入，所以 Mel 频谱非常合适
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=64
        )

    def __len__(self):
        return len(self.data_frame)

    def _get_wav_path(self, filename):
        # 根据 IEMOCAP 的命名规则还原 .wav 文件的物理路径
        # filename 例如: "Ses01F_impro01_F000"
        session = f"Session{filename[4]}"  # 提取数字变成 "Session1"
        dialog_name = filename.rsplit('_', 1)[0]  # 提取变成 "Ses01F_impro01"

        wav_path = os.path.join(
            self.iemocap_root,
            session,
            "sentences",
            "wav",
            dialog_name,
            f"{filename}.wav"
        )
        return wav_path

    def __getitem__(self, idx):
        # 1. 获取文件名和情感标签
        filename = self.data_frame.iloc[idx, 0]
        emotion_str = self.data_frame.iloc[idx, 1]

        # 2. 转换标签为整数
        label = self.emotion_to_idx[emotion_str]

        # 3. 构建音频路径并加载音频
        wav_path = self._get_wav_path(filename)
        waveform_np, sample_rate = sf.read(wav_path)

        waveform = torch.from_numpy(waveform_np).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # [1, 时间长度]

        # 统一采样率
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        # 4. 提取 Mel 频谱
        mel_spec = self.mel_spectrogram(waveform)  # [1, 64, T]
        mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)

        # ---------------- 新增：计算 Δ 和 ΔΔ ----------------
        def compute_delta(feat, N=2):
            """
            feat: [num_mels, time]
            """
            T = feat.shape[1]
            denominator = 2 * sum([i ** 2 for i in range(1, N + 1)])
            delta_feat = torch.zeros_like(feat)

            for t in range(T):
                numerator = 0
                for n in range(1, N + 1):
                    t_plus = min(T - 1, t + n)
                    t_minus = max(0, t - n)
                    numerator += n * (feat[:, t_plus] - feat[:, t_minus])
                delta_feat[:, t] = numerator / denominator

            return delta_feat

        mel_spec = mel_spec.squeeze(0)  # [64, T]
        delta = compute_delta(mel_spec)
        delta_delta = compute_delta(delta)

        # 5. 堆叠成三通道 [3, 64, T]
        mel_spec = torch.stack([mel_spec, delta, delta_delta], dim=0)

        if self.transform:
            mel_spec = self.transform(mel_spec)

        return mel_spec, label


import torch.nn.functional as F
from torch.utils.data import random_split


import torch
import torch.nn.functional as F

def pad_collate(batch):
    """
    batch: 包含多个元组 (mel_spec, label) 的列表
    现在 mel_spec 是三通道 [3, 64, T]
    """
    mels, labels = zip(*batch)

    # 固定时间帧长度 (可根据大部分音频长度调整)
    FIXED_LEN = 256

    padded_mels = []
    for mel in mels:
        # mel.shape: [3, 64, T]
        T = mel.shape[2]
        if T < FIXED_LEN:
            # 时间维度右侧补零
            pad_amount = FIXED_LEN - T
            # F.pad 参数格式: (左, 右, 上, 下, ...)
            # 对 3D tensor, pad 对最后两个维度起作用: (W_pad_left, W_pad_right, H_pad_top, H_pad_bottom)
            padded_mel = F.pad(mel, (0, pad_amount, 0, 0))
        else:
            # 超长直接截断
            padded_mel = mel[:, :, :FIXED_LEN]
        padded_mels.append(padded_mel)

    # 堆叠成 batch
    mels_batch = torch.stack(padded_mels)       # [B, 3, 64, FIXED_LEN]
    labels_batch = torch.tensor(labels, dtype=torch.long)

    return mels_batch, labels_batch


# ================= 数据集划分与加载测试 =================
if __name__ == "__main__":
    CSV_PATH = "D:\mobile_audio\DataSet\IEMOCAP_Labels\my_iemocap_labels_6class.csv"
    IEMOCAP_ROOT = r"D:\mobile_audio\DataSet\IEMOCAP_full_release"

    # 1. 实例化数据集
    full_dataset = IEMOCAPDataset(csv_file=CSV_PATH, iemocap_root=IEMOCAP_ROOT)

    # 2. 划分训练集 (80%) 和 测试集 (20%)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    print(f"训练集大小: {len(train_dataset)} | 测试集大小: {len(test_dataset)}")

    # 3. 创建 DataLoader (指定 batch_size 和 我们刚写的 pad_collate 函数)
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=pad_collate,
        num_workers=4,
        pin_memory=True
    )

