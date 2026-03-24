import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class AudioEmotionDataset(Dataset):
    def __init__(self, data_info: pd.DataFrame, label_mapping: dict, npz_root: str, is_train=True):
        """
        自定义语音情感数据集
        :param data_info: 包含 Filename 和 Emotion 列的 DataFrame (由 read_file.py 提供)
        :param label_mapping: 情感标签到数字 ID 的映射字典
        :param npz_root: 预处理好的 .npz 特征文件存放目录
        :param is_train: 是否为训练集（未来可用于控制数据增强）
        """
        self.data_info = data_info
        self.label_mapping = label_mapping
        self.npz_root = npz_root
        self.is_train = is_train

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        # 1. 获取当前数据的文件名和标签
        file_id = str(self.data_info.loc[idx, 'Filename'])
        emotion_str = self.data_info.loc[idx, 'Emotion']
        label = self.label_mapping[emotion_str]

        # 2. 拼接 .npz 文件的绝对路径
        npz_path = os.path.join(self.npz_root, f"{file_id}.npz")

        try:
            # 读取打包好的特征
            data = np.load(npz_path)
            mel_spec = data['mel_spectrogram']  # 形状期望是 (3, 224, 224)
            wavlm_feat = data['wavlm']  # 形状期望是 (768,)
        except Exception as e:
            print(f"读取异常文件: {npz_path}, 错误: {e}")
            # 容错处理：如果文件损坏或丢失，返回全零张量防止训练崩溃
            mel_spec = np.zeros((3, 224, 224), dtype=np.float32)
            wavlm_feat = np.zeros((768,), dtype=np.float32)

        # 3. 将 numpy 数组转换为 PyTorch 要求的 Tensor 格式
        # 注意这里必须是 FloatTensor，对应模型的 float32 权重
        mel_tensor = torch.FloatTensor(mel_spec)
        wavlm_tensor = torch.FloatTensor(wavlm_feat)
        label_tensor = torch.tensor(label, dtype=torch.long)

        # 返回：视觉特征(主), 听觉特征(辅), 真实标签
        return mel_tensor, wavlm_tensor, label_tensor


def get_loader(data_info, label_mapping, npz_root, batch_size=16, is_train=True):
    """
    创建并返回 DataLoader，负责将数据按 Batch 打包送入 GPU
    """
    dataset = AudioEmotionDataset(data_info, label_mapping, npz_root, is_train)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_train,  # 训练时打乱顺序，测试时不打乱
        num_workers=0,  # Windows 下建议先设为 0 防止多进程报错，跑通后可尝试改为 4 提速
        pin_memory=True,  # 开启此选项可以稍微加速数据向 GPU 的搬运
        drop_last=False  # 不丢弃最后一个不完整的 batch
    )
    return loader


# ==================== 测试代码 ====================
if __name__ == "__main__":
    from read_file import read_csv

    # ！！！请把下面的路径换成你刚才跑通预处理时的真实路径！！！
    TEST_CSV = r"D:\mobile_audio\DataSet\Data_IEMOCAP\IEMOCAP_Labels\session1_labels.csv"
    TEST_NPZ = r"D:\mobile_audio\DataSet\Data_IEMOCAP\processed_features_iemocap"  # 你存放 0.npz, 1.npz 的那个文件夹

    print("开始测试 DataLoader...")
    data, mapping = read_csv(TEST_CSV)

    # 模拟创建一个 batch size 为 4 的加载器
    test_loader = get_loader(data, mapping, TEST_NPZ, batch_size=4)

    # 抽取第一个 Batch 看看长什么样
    for mel, wavlm, label in test_loader:
        print("\n--- 成功抓取一个 Batch 的数据 ---")
        print(f"梅尔频谱 (MobileViT的输入) Shape: {mel.shape}")  # 期望输出: torch.Size([4, 3, 224, 224])
        print(f"WavLM 特征 (辅助分支的输入) Shape: {wavlm.shape}")  # 期望输出: torch.Size([4,