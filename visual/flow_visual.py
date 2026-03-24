import numpy as np
import cv2
import matplotlib.pyplot as plt

def normalize_flow(flow_component):
    # 将光流分量归一化到0到255
    normalized = cv2.normalize(flow_component, None, 0, 255, cv2.NORM_MINMAX)
    normalized = np.uint8(normalized)
    return normalized

def TVL1_optical_flow(prev_frame: np.array, next_frame: np.array):
    # 将输入的前一帧和后一帧图像从RGB色彩空间转换为灰度图像
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    next_frame = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
    # Create TV-L1 optical flow
    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create(scaleStep=0.5)
    flow = optical_flow.calc(prev_frame, next_frame, None)
    return flow

# 读取视频或两帧图像
prev_frame = cv2.imread('F:\PM_success\moblie_repvit\moblie_repvit\PyTorch-DSSN-NEW-main\img46.jpg')
next_frame = cv2.imread('F:\PM_success\moblie_repvit\moblie_repvit\PyTorch-DSSN-NEW-main\img59.jpg')

# 计算光流
flow = TVL1_optical_flow(prev_frame, next_frame)

# 提取水平和垂直光流分量
horizontal_flow = flow[..., 0]
vertical_flow = flow[..., 1]

# 归一化光流分量
normalized_horizontal = normalize_flow(horizontal_flow)
normalized_vertical = normalize_flow(vertical_flow)

# 显示图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(normalized_horizontal, cmap='gray')
plt.title('Horizontal Optical Flow')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(normalized_vertical, cmap='gray')
plt.title('Vertical Optical Flow')
plt.colorbar()

plt.show()
