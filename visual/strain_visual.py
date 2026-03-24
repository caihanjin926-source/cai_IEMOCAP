import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_optical_flow(img1, img2):
    # 将图像转换为灰度
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 创建TV-L1光流对象
    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()

    # 计算光流
    flow = optical_flow.calc(gray1, gray2, None)
    return flow


def calculate_strain(flow):
    # 计算流场的梯度
    flow_grad_x = np.gradient(flow[:, :, 0], axis=0)
    flow_grad_y = np.gradient(flow[:, :, 1], axis=1)

    # 计算应变
    strain = np.sqrt(flow_grad_x ** 2 + flow_grad_y ** 2)
    return strain


def plot_strain(strain):
    # 显示应变图
    plt.imshow(strain, cmap='hot')
    plt.colorbar()
    plt.title('Optical Strain')
    plt.show()


# 加载两帧图像
img1 = cv2.imread('F:\PM_success\moblie_repvit\moblie_repvit\PyTorch-DSSN-NEW-main\img46.jpg')
img2 = cv2.imread('F:\PM_success\moblie_repvit\moblie_repvit\PyTorch-DSSN-NEW-main\img59.jpg')

# 计算光流
flow = calculate_optical_flow(img1, img2)

# 计算应变
strain = calculate_strain(flow)

# 可视化应变
plot_strain(strain)
