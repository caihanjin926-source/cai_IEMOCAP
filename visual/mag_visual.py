import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy

def calculate_optical_flow(img1, img2):
    # 将图像转换为灰度
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 创建TV-L1光流对象
    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()

    # 计算光流
    flow = optical_flow.calc(gray1, gray2, None)
    return flow

def TVL1_magnitude(flow: np.array):
    # 创建光流场的深层副本，以确保对原始数据没有影响
    flow = copy.deepcopy(flow)
    # 计算光流场每个像素点的大小（magnitude）
    mag = np.sqrt(np.sum(flow ** 2, axis=-1))
    # 调用了一个名为normalized_channel的函数对计算得到的大小进行标准化处理
    mag = normalized_channel(mag)
    return mag

def normalized_channel(channel):
    # 将通道数值标准化到0-1范围
    return (channel - channel.min()) / (channel.max() - channel.min())

def plot_magnitude(mag):
    # 使用viridis颜色映射可视化光流场大小
    plt.imshow(mag, cmap='viridis')
    plt.colorbar()
    plt.title('Optical Flow Magnitude')
    plt.show()



# 加载两帧图像
img1 = cv2.imread('F:\PM_success\moblie_repvit\moblie_repvit\PyTorch-DSSN-NEW-main\img46.jpg')
img2 = cv2.imread('F:\PM_success\moblie_repvit\moblie_repvit\PyTorch-DSSN-NEW-main\img59.jpg')
# 计算光流
flow = calculate_optical_flow(img1, img2)

# 计算光流场大小
mag = TVL1_magnitude(flow)

# 可视化光流场大小
plot_magnitude(mag)
