import torch
import torch.nn as nn
import timm


class AudioEmotionFusionNet(nn.Module):
    def __init__(self, num_classes=4, freeze_mobilevit=True):
        """
        双分支情感识别网络：MobileViT 处理梅尔频谱 + Linear 处理 WavLM 特征
        :param num_classes: 分类数 (IEMOCAP 默认 4 类)
        """
        super(AudioEmotionFusionNet, self).__init__()

        # ================= 分支 1：MobileViT =================
        # 使用 timm 一键加载 MobileViT-XXS！
        # pretrained=True: 会自动从服务器下载官方预训练权重（比本地加载更稳）
        # num_classes=0: 表示我们不需要它的分类头，只要它提取出来的 320 维特征向量
        print("正在初始化 MobileViT...")
        self.mobilevit = timm.create_model('mobilevit_xxs', pretrained=True, num_classes=0)

        # 冻结底层的特征提取器，加速训练并防止小数据集过拟合
        if freeze_mobilevit:
            print("已冻结 MobileViT 底层权重。")
            for param in self.mobilevit.parameters():
                param.requires_grad = False

        mobilevit_out_features = 320  # XXS 版本的输出维度固定是 320

        # ================= 分支 2：处理 WavLM 特征 =================
        wavlm_in_features = 768
        self.wavlm_branch = nn.Sequential(
            nn.Linear(wavlm_in_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)  # 降维到 128
        )

        # ================= 特征融合层与分类器 =================
        # 将两路特征拼接起来：320 + 128 = 448
        fusion_features = mobilevit_out_features + 128

        self.classifier = nn.Sequential(
            nn.Linear(fusion_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, mel_spec, wavlm_feat):
        """
        前向传播
        """
        # 分支 1：视觉特征提取
        img_features = self.mobilevit(mel_spec)
        # 安全检查：如果输出是4维的 (B, C, H, W)，就做一次全局平均池化变成 (B, C)
        if len(img_features.shape) == 4:
            img_features = img_features.mean([2, 3])

            # 分支 2：听觉特征降维
        audio_features = self.wavlm_branch(wavlm_feat)

        # 融合与分类
        fused = torch.cat((img_features, audio_features), dim=1)
        logits = self.classifier(fused)

        return logits


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("测试基于 timm 的网络架构...")

    dummy_model = AudioEmotionFusionNet(num_classes=4)

    # 模拟 DataLoader 吐出的数据
    dummy_mel = torch.randn(4, 3, 224, 224)
    dummy_wavlm = torch.randn(4, 768)

    # 传入网络
    out = dummy_model(dummy_mel, dummy_wavlm)
    print(f"\n✅ 融合网络前向传播成功！输出形状: {out.shape} (期望是 [4, 4])")