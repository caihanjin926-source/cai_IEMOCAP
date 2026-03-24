import matplotlib.pyplot as plt
import numpy as np

# ================= 提取的实验数据 =================
epochs = np.arange(1, 21)

# 实验A：Mel + ImageNet预训练 + SpecAugment (最高 44.82%)
train_loss_A = [1.6609, 1.5115, 1.4493, 1.4063, 1.3666, 1.3347, 1.2976, 1.2831, 1.2356, 1.1938, 1.1663, 1.1258, 1.0869, 1.0511, 0.9961, 0.9684, 0.9220, 0.8840, 0.8409, 0.7994]
val_loss_A = [1.5495, 1.4681, 1.4290, 1.4368, 1.4021, 1.4272, 1.4624, 1.4079, 1.4128, 1.4318, 1.4217, 1.4622, 1.5004, 1.5211, 1.5591, 1.6236, 1.6426, 1.7064, 1.7274, 1.7225]
val_acc_A = [38.09, 41.00, 42.63, 40.90, 43.29, 43.29, 42.38, 43.75, 44.82, 43.75, 43.96, 42.63, 42.94, 42.38, 41.66, 42.07, 42.99, 42.53, 42.07, 41.41]

# 实验B：WavLM 深层特征无增强 (最高 40.54%)
train_loss_B = [1.6930, 1.5740, 1.4944, 1.4166, 1.3437, 1.2607, 1.1861, 1.0994, 1.0147, 0.9289, 0.8411, 0.7810, 0.6953, 0.6206, 0.5617, 0.5102, 0.4562, 0.4205, 0.3604, 0.3255]
val_loss_B = [1.6004, 1.6082, 1.4856, 1.5341, 1.5315, 1.5309, 1.5416, 1.5943, 1.6389, 1.8213, 1.9355, 1.9978, 2.0863, 2.2387, 2.4638, 2.4294, 2.4017, 2.4968, 2.4899, 2.6750]
val_acc_B = [36.77, 33.50, 40.54, 39.11, 39.72, 38.81, 40.39, 39.93, 40.03, 37.58, 37.48, 37.28, 38.96, 36.46, 35.03, 36.36, 37.33, 37.07, 36.21, 36.51]

# ================= 开始绘图 =================
plt.style.use('ggplot') # 使用美观的样式
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# ---- 图1: 训练集 vs 验证集 Loss 曲线对比 ----
ax1.plot(epochs, train_loss_A, 'b-', label='Exp 1: Mel + SpecAugment (Train)')
ax1.plot(epochs, val_loss_A, 'b--', linewidth=2, label='Exp 1: Mel + SpecAugment (Val)')
ax1.plot(epochs, train_loss_B, 'r-', label='Exp 2: WavLM Feature (Train)')
ax1.plot(epochs, val_loss_B, 'r--', linewidth=2, label='Exp 2: WavLM Feature (Val)')

ax1.set_title('Training & Validation Loss Comparison', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epochs', fontsize=12)
ax1.set_ylabel('Cross Entropy Loss', fontsize=12)
ax1.set_xticks(range(2, 21, 2))
ax1.legend(loc='best')

# ---- 图2: 验证集准确率 (Val Accuracy) 曲线对比 ----
ax2.plot(epochs, val_acc_A, 'b-o', markersize=6, linewidth=2, label='Exp 1 (Max: 44.82%)')
ax2.plot(epochs, val_acc_B, 'r-s', markersize=6, linewidth=2, label='Exp 2 (Max: 40.54%)')

# 标注最高点
max_idx_A = np.argmax(val_acc_A)
max_idx_B = np.argmax(val_acc_B)
ax2.annotate(f'44.8%', xy=(max_idx_A+1, val_acc_A[max_idx_A]), xytext=(0, 10), textcoords='offset points', ha='center', color='blue', fontweight='bold')
ax2.annotate(f'40.5%', xy=(max_idx_B+1, val_acc_B[max_idx_B]), xytext=(0, 10), textcoords='offset points', ha='center', color='red', fontweight='bold')

ax2.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epochs', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_xticks(range(2, 21, 2))
ax2.legend(loc='lower right')

plt.tight_layout()
plt.savefig('presentation_plot.png', dpi=300) # 保存为高清图片
print("✅ 高清对比图已成功保存为: presentation_plot.png")
plt.show() # 在弹窗中显示