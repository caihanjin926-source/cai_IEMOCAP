import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from preprocess import compute_audio_features
from read_file import read_csv


def build_file_map(audio_root):
    """遍历所有子目录，找到所有的 mp4 和 wav 文件，建立 名字->绝对路径 的字典"""
    print(f"正在扫描 {audio_root} 下的所有音频/视频文件，请稍候...")
    file_map = {}
    for root, _, files in os.walk(audio_root):
        for f in files:
            if f.endswith('.wav') or f.endswith('.mp4'):
                # 去掉后缀，拿到纯文件名 (例如 dia0_utt0)
                file_id = os.path.splitext(f)[0]
                file_map[file_id] = os.path.join(root, f)
    print(f"扫描完毕，共找到 {len(file_map)} 个文件！")
    return file_map


def process_and_save_audio(data_info: pd.DataFrame, file_map: dict, save_root: str, wavlm_root: str = ""):
    os.makedirs(save_root, exist_ok=True)

    # 记录没找到的文件，方便排查
    not_found_list = []

    for idx in tqdm(range(len(data_info))):
        file_id = data_info.loc[idx, 'Filename']

        # 从刚才建立的字典里直接拿文件的绝对路径
        audio_path = file_map.get(file_id)

        if audio_path is None:
            not_found_list.append(file_id)
            continue

        # 1. 计算三通道梅尔频谱图特征 (librosa 会自动从 mp4 抽音频)
        try:
            mel_tensor = compute_audio_features(audio_path)
        except Exception as e:
            print(f"读取 {audio_path} 失败，可能文件损坏或缺少 ffmpeg，报错: {e}")
            continue

        # 2. WavLM 特征占位
        wavlm_feat = np.zeros((768,))
        if wavlm_root and os.path.exists(os.path.join(wavlm_root, f"{file_id}.npy")):
            wavlm_feat = np.load(os.path.join(wavlm_root, f"{file_id}.npy"))

        # 3. 保存为 .npz 文件
        save_path = os.path.join(save_root, f"{file_id}.npz")
        np.savez(
            file=save_path,
            mel_spectrogram=mel_tensor,
            wavlm=wavlm_feat
        )

    if not_found_list:
        print(f"\n警告：CSV中有 {len(not_found_list)} 个文件在文件夹中未找到对应的 wav/mp4。")
        print(f"前5个未找到的例子: {not_found_list[:5]}")


if __name__ == "__main__":
    # ================== 请修改这里的路径 ==================
    # 1. 替换成你真实的 CSV 绝对路径
    # 注意：你需要具体到某一个 csv 文件，例如 session1_labels.csv
    CSV_PATH = r"D:\mobile_audio\DataSet\Data_IEMOCAP\IEMOCAP_Labels\session1_labels.csv"

    # 2. 替换成你真实的包含 IEMOCAP 语音的文件夹路径
    # 例如 IEMOCAP 官方数据集下的 Session1 的 wav 文件夹
    AUDIO_ROOT = r"D:\mobile_audio\DataSet\Data_IEMOCAP\Session1\sentences\wav"  # 请根据你电脑的实际路径修改！

    # 3. 这个是可以自己随便定一个文件夹存提取好的特征的
    # 建议加上数据集名字以便区分
    SAVE_ROOT = r"D:\mobile_audio\processed_features\IEMOCAP_npz\Session1"

    WAVLM_ROOT = ""
    # ======================================================

    # 读取标签
    data, mapping = read_csv(CSV_PATH)

    # 扫描文件夹，建立映射
    file_map = build_file_map(AUDIO_ROOT)

    # 开始提取
    print("开始提取音频特征并保存...")
    process_and_save_audio(data, file_map, SAVE_ROOT, WAVLM_ROOT)
    print("特征提取全部完成！")