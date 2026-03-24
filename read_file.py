import numpy as np
import pandas as pd


def read_csv(path: str) -> tuple:
    # 尝试读取 CSV (header=None 专门为了防止把第一行数字当成表头)
    try:
        data = pd.read_csv(path, header=None)
    except Exception as e:
        raise ValueError(f"读取 CSV 失败: {e}")

    # ================= 情况 A：只有一列数字的纯标签 CSV =================
    if len(data.columns) == 1:
        print(f"\n[提示] 检测到纯标签文件，自动按 '0', '1', '2'... 分配文件名。")
        data.columns = ['Emotion']
        # 强行给每一行分配一个索引名 "0", "1", "2"... 作为文件名，以对齐你提取的 .npz
        data['Filename'] = data.index.astype(str)

    # ================= 情况 B：MELD 官方完整 CSV =================
    elif 'Dialogue_ID' in data.columns and 'Utterance_ID' in data.columns and 'Emotion' in data.columns:
        print("检测到 MELD 官方 CSV 格式...")
        # 重新读取，带有表头
        data = pd.read_csv(path)
        data['Filename'] = "dia" + data['Dialogue_ID'].astype(str) + "_utt" + data['Utterance_ID'].astype(str)

    # ================= 情况 C：通用带表头的 CSV =================
    else:
        data = pd.read_csv(path)
        filename_col = None
        for col in ['Filename', 'name', 'file', 'wav_file', 'Utterance', 'Session_ID']:
            if col in data.columns: filename_col = col; break
        emotion_col = None
        for col in ['Emotion', 'label', 'emotion', 'Label']:
            if col in data.columns: emotion_col = col; break

        if filename_col is None or emotion_col is None:
            raise ValueError(f"无法找到文件名或情感列。现有列: {data.columns.tolist()}")

        data = data.rename(columns={filename_col: 'Filename', emotion_col: 'Emotion'})
        data['Filename'] = data['Filename'].apply(lambda x: str(x).split('.')[0].strip())

    # 建立情感映射字典
    unique_emotions = np.unique(data['Emotion'].dropna())
    label_mapping = {emotion: idx for idx, emotion in enumerate(unique_emotions)}
    print(f"情感标签映射表: {label_mapping}")

    return data, label_mapping