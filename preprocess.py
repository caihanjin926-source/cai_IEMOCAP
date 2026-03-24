import librosa
import numpy as np
import cv2


def compute_audio_features(audio_path, target_size=(224, 224)):
    """
    读取音频并生成适用于 MobileViT 的 3通道梅尔频谱图
    """
    # 1. 加载音频，统一采样率为 16000Hz
    y, sr = librosa.load(audio_path, sr=16000)

    # 2. 提取梅尔频谱图
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    # 3. 计算一阶和二阶差分
    delta = librosa.feature.delta(log_mel)
    delta2 = librosa.feature.delta(log_mel, order=2)

    # 4. 堆叠成 (H, W, C) 格式
    mel_img = np.stack([log_mel, delta, delta2], axis=-1)

    # 5. Resize 为 MobileViT 需要的尺寸 (224x224)
    mel_resized = cv2.resize(mel_img, target_size)

    # 6. 维度转换 (H, W, C) -> (C, H, W)，即 (3, 224, 224)
    mel_tensor = np.transpose(mel_resized, (2, 0, 1))

    return mel_tensor