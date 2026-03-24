import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class WavLMDataset(Dataset):
    def __init__(self, csv_file, feature_dir):
        """
        直接读取离线提取好的 WavLM .pt 特征文件
        """
        self.data_frame = pd.read_csv(csv_file)
        self.feature_dir = feature_dir

        self.emotion_to_idx = {
            'ang': 0, 'exc': 1, 'fru': 2,
            'hap': 3, 'neu': 4, 'sad': 5
        }

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        filename = self.data_frame.iloc[idx, 0]
        emotion_str = self.data_frame.iloc[idx, 1]
        label = self.emotion_to_idx[emotion_str]

        # 1. 直接加载 .pt 文件 (形状已经是 [768, Time])
        pt_path = os.path.join(self.feature_dir, f"{filename}.pt")
        feature = torch.load(pt_path)

        # 2. 伪装成 3 通道图像 [3, 768, Time] 以便使用 ImageNet 预训练权重
        feature = feature.unsqueeze(0).repeat(3, 1, 1)

        return feature, label


# 专属的数据对齐函数
def pad_collate_wavlm(batch):
    features, labels = zip(*batch)

    # WavLM 的输出帧率大约是 50 帧/秒。256 帧大约相当于 5.1 秒的音频。
    # 这个长度覆盖了 IEMOCAP 中绝大多数的对话句子。
    FIXED_LEN = 256

    padded_features = []
    for feat in features:
        T = feat.shape[2]
        if T < FIXED_LEN:
            pad_amount = FIXED_LEN - T
            padded_feat = F.pad(feat, (0, pad_amount))
        else:
            padded_feat = feat[:, :, :FIXED_LEN]
        padded_features.append(padded_feat)

    features_batch = torch.stack(padded_features)
    labels_batch = torch.tensor(labels, dtype=torch.long)

    return features_batch, labels_batch