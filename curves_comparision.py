import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']     # 显示中文
# 为了坐标轴负号正常显示。matplotlib默认不支持中文，设置中文字体后，负号会显示异常。需要手动将坐标轴负号设为False才能正常显示负号。
matplotlib.rcParams['axes.unicode_minus'] = False

# 假设这些是三个模型在训练过程中收集的损失和准确率数据
losses_model1 = [0.9825, 0.7190, 0.5912,0.2748, 0.2615,0.2413,0.2821,0.1409,0.0828,0.07,0.1352,0.0916,0.0934,0.0332,0.1826,0.1271,0.1065,0.0855,0.0546, 0.0432]
losses_model2 = [0.9513, 0.7988, 0.5598,0.3101, 0.1931,0.1514,0.1832,0.1664,0.1213,0.1222,0.0792,0.0869,0.0932,0.0234,0.1610, 0.0475,0.0237,0.0414,0.0228,0.0148]

accuracies_model1 = [0.5350,0.7390,0.7785,0.9188,0.9035,0.9300,0.9013,0.9405,0.9671,0.9802,0.9627,0.9736, 0.9671,0.9782, 0.9298,0.9605, 0.9627,0.9671,0.9736,0.9762]
accuracies_model2 = [0.5175,0.6842,0.8114,0.9276,0.9473,0.9539,0.9473,0.9364,0.9539,0.9627,0.9802, 0.9771, 0.9661,0.9934,0.9561, 0.9868,0.9934,0.9868,0.9934,0.9836]


# 创建一个图形帧，设置大小
plt.figure(figsize=(10, 10),dpi=600)

# 绘制三个模型的损失曲线对比
plt.subplot(2, 1, 1)  # 2行1列的第1个
plt.plot(losses_model1, label='Dual-stream network', linestyle='-', color='red')
plt.plot(losses_model2, label='Dual-stream network + TC', linestyle='--', color='blue')

plt.title('Comparison of Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 绘制三个模型的准确率曲线对比
plt.subplot(2, 1, 2)  # 2行1列的第2个
plt.plot(accuracies_model1, label='Dual-stream network', linestyle='-', color='red')
plt.plot(accuracies_model2, label='Dual-stream network + TC', linestyle='--', color='blue')

plt.title('Comparison of Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 显示图形
plt.show()
