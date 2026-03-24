# 导入matplotlib库
import matplotlib.pyplot as plt

# 模型 1 和 模型 2 的数据
M1 = [4.94, 2.32, 0.93]  # 参数量 (模型 1)
Flops1 = [972.45, 561.12, 206.13]  # 浮点运算量 (模型 1)

M2 = [4.95, 2.32, 0.933]  # 参数量 (模型 2)
Flops2 = [920.67, 512.9, 168.57]  # 浮点运算量 (模型 2)

# 绘制模型 1 的折线图
plt.plot(M1, Flops1, label="MobileViT", marker='o', linestyle='-', color='b')

# 绘制模型 2 的折线图
plt.plot(M2, Flops2, label="MobileViT with LDA", marker='s', linestyle='--', color='g')

# 添加标题和轴标签
# plt.title('Comparison of Model 1 and Model 2')
plt.xlabel('Parameter (M)')
plt.ylabel('FLOPS (M)')

# 显示图例
plt.legend()

# 显示网格
plt.grid(True)

# 展示图表
plt.show()
