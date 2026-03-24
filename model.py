import torch
import torch.nn as nn
import timm


class AudioMobileViT(nn.Module):
    def __init__(self, num_classes=6):
        super(AudioMobileViT, self).__init__()

        # 关键修改：开启预训练 (pretrained=True)，输入通道改为 3 (in_chans=3)
        self.backbone = timm.create_model(
            'mobilevit_xs',
            pretrained=True,  # <--- 白嫖几千万图像学到的特征提取能力
            in_chans=3,  # <--- 准备接收伪装成 RGB 的频谱图
            num_classes=num_classes
        )

    def forward(self, x):
        # x 现在的形状期望是: [Batch_size, 3, 64, 256]
        return self.backbone(x)

# ================= 测试模型结构 =================
if __name__ == "__main__":
    # 实例化模型
    model = AudioMobileViT(num_classes=6)

    # 打印模型参数量，心里有个底
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MobileViT 模型可训练参数总量: {total_params / 1e6:.2f} M (百万)")

    # 模拟一个 DataLoader 吐出来的 Batch 数据
    # 形状: [batch_size=16, channel=1, freq=64, time=256]
    dummy_input = torch.randn(16, 1, 64, 256)

    # 将模拟数据送入模型
    output = model(dummy_input)

    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape} <-- 期望看到 [16, 6] (16个样本，每个样本6个情感类别的得分)")