import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def confusion_matrix():

    data_CASME = np.array([[0.86, 0.09, 0.05],
                     [0.17, 0.72, 0.11],
                     [0.06, 0.1, 0.84]])
    data_SAMM = np.array([[0.76, 0.15, 0.09],
                     [0.34, 0.62, 0.04],
                     [0.21, 0.09, 0.7]])
    data_CASME2= np.array([[0.85, 0.1, 0.05],
                     [0.16, 0.72, 0.12],
                     [0.07, 0.11, 0.82]])
    data_SAMM2 = np.array([[0.76, 0.14, 0.1],
                          [0.31, 0.6, 0.09],
                          [0.18, 0.14, 0.68]])
    # 标签
    labels = ['Negative', 'Positive', 'Surprise']

    fig, ax = plt.subplots()
    # 创建热图
    cax = ax.matshow(data_SAMM2, cmap='Blues', norm=Normalize(vmin=0, vmax=1))

    # 设置轴标签
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # 设置轴标题
    ax.set_xlabel('Predicted Label', labelpad=10)
    ax.xaxis.set_label_position('bottom')
    ax.set_ylabel('True Label')

    # 设置每个单元格的标签
    for (i, j), val in np.ndenumerate(data_SAMM2):
        color = 'white' if i == j else 'black'  # 对角线为白色，其他为黑色
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color)

    # 旁边的颜色梯度对比
    plt.colorbar(cax)

    plt.show()

confusion_matrix()