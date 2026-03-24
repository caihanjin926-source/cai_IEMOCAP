import os
import torch
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

# ================= 配置路径 =================
CSV_PATH = "D:\mobile_audio\DataSet\IEMOCAP_Labels\my_iemocap_labels_6class.csv"
IEMOCAP_ROOT = r"D:\mobile_audio\DataSet\IEMOCAP_full_release"
OUTPUT_DIR = r"D:\mobile_audio\DataSet\wavlm_features"  # 存放提取好的特征的文件夹


def get_wav_path(filename, root):
    """根据 IEMOCAP 命名规则拼接音频路径"""
    session = f"Session{filename[4]}"
    dialog_name = filename.rsplit('_', 1)[0]
    return os.path.join(root, session, "sentences", "wav", dialog_name, f"{filename}.wav")


def main():
    # 1. 准备设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 使用计算设备: {device}")

    # 创建输出文件夹
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. 加载 WavLM 模型 (使用 base-plus 版本，效果好且适合单卡)
    print("⏳ 正在下载/加载 WavLM 模型 (初次运行可能需要一点时间)...")
    # WavLM 使用和 Wav2Vec2 相同的特征处理器
    processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device)
    model.eval()  # 设为评估模式

    # 3. 读取 CSV 文件
    df = pd.read_csv(CSV_PATH)
    print(f"✅ 成功读取标签文件，共 {len(df)} 条数据。开始提取特征...")

    # 4. 遍历所有文件，提取特征并保存
    with torch.no_grad():  # 不计算梯度，节省显存加速提取
        for idx in tqdm(range(len(df)), desc="提取 WavLM 特征"):
            filename = df.iloc[idx, 0]
            wav_path = get_wav_path(filename, IEMOCAP_ROOT)

            # 目标保存路径: D:\mobile_audio\wavlm_features\Ses01F_impro01_F000.pt
            save_path = os.path.join(OUTPUT_DIR, f"{filename}.pt")

            # 如果已经提取过，就跳过 (方便中断后断点续传)
            if os.path.exists(save_path):
                continue

            try:
                # 读取音频
                waveform, sample_rate = sf.read(wav_path)

                # WavLM 强烈要求 16000Hz 采样率 (IEMOCAP 本身就是 16k)
                inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
                input_values = inputs.input_values.to(device)

                # 送入模型提取深层特征
                outputs = model(input_values)

                # 获取最后一层隐状态: 形状为 [1, Time, 768]
                hidden_states = outputs.last_hidden_state

                # 转换形状为 [768, Time]，为了后续当作有 768 个通道的“特征图”喂给 MobileViT
                feature = hidden_states.squeeze(0).transpose(0, 1)  # 变成 [768, Time]

                # 转移回 CPU 并保存到硬盘
                torch.save(feature.cpu(), save_path)

            except Exception as e:
                print(f"\n❌ 处理 {filename} 时出错: {e}")

    print(f"\n🎉 全部特征提取完成！特征保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()