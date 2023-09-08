
import numpy
import torch
from prettytable import PrettyTable
from matplotlib.colors import ListedColormap
# import numpy as np
# from timm.models.layers import to_2tuple #MLP调用
# 计算评价指标，并保存为表格
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.family'] = 'SimSun' # 设置字体为宋体

import numpy as np
np.set_printoptions(suppress=True)

def get_matrix():
    name = "convnext_2.pth"
    # maps = torch.load("../weight/save/{}".format(name))
    # maps = torch.load("../weight/save/convnext 22k 90.79 drop 0.1/convnext_22k.pth")
    maps = torch.load(basepath)
    # "weight/save/convnext_zigong.pth"
    print(maps["epoch"])
    print(maps["best_acc"])
    print(maps["best_matrix"])
    return name, maps["best_acc"], maps["best_matrix"]

def summary(savepath):
    # precision, recall, specificity, f1_score
    name, acc, matrix = get_matrix()
    # "Acne", "Melasma", "Rosacea", "Discoid lupus erythematosus", "Ota nevus", "Seborrheic dermatitis"
    # "AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"
    # ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
    label = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
    # label = ['AC', 'DLE', 'ME', 'ON', 'ROS', 'SD']
    # label = ['cancer', 'natural']
    class_num = len(label)
    table = PrettyTable()
    table.field_names = ["", "Precision", "Recall", "Specificity", "F1-Score", "FNR"]
    with open(savepath, "a", encoding="utf-8") as file:
        content = "label,Precision,Recall,Specificity,F1-Score,{},{}\n".format(name, acc)
        file.write(content)
    for i in range(class_num):
        TP = matrix[i, i]
        FP = np.sum(matrix[i, :]) - TP
        FN = np.sum(matrix[:, i]) - TP
        TN = np.sum(matrix) - TP - FP - FN
        precision = round(TP / (TP + FP), 4) if TP + FP != 0 else 0.
        recall = round(TP / (TP + FN), 4) if TP + FN != 0 else 0.
        specificity = round(TN / (TN + FP), 4) if TN + FP != 0 else 0.
        f1_score = round(2 * (precision * recall) / (precision + recall), 4)
        FNR = round(FN / (TP + FN), 4) if TP + FN != 0 else 0.
        table.add_row([label[i], precision, recall, specificity, f1_score, FNR])
        with open(savepath, "a", encoding="utf-8") as file:
            content = "{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(label[i], precision, recall, specificity, f1_score, FNR)
            file.write(content)
    print(table)

def draw(confusion_matrix,metrixpath):
    # # 定义类别标签
    # labels = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
    #
    # # 创建图像和轴对象
    # fig, ax = plt.subplots(figsize=(8, 8))
    #
    # # 画出混淆矩阵
    # im = ax.imshow(confusion_matrix, cmap='Oranges')
    #
    # # 设置轴标签和刻度
    # ax.set_xticks(np.arange(len(labels)))
    # ax.set_yticks(np.arange(len(labels)))
    # ax.set_xticklabels(labels, fontsize=12)
    # ax.set_yticklabels(labels, fontsize=12)
    # ax.set_xlabel('True class', fontsize=14)
    # ax.set_ylabel('Predicted class', fontsize=14)
    #
    # # 在每个单元格中添加文本注释
    # for i in range(len(labels)):
    #     for j in range(len(labels)):
    #         text = ax.text(j, i, confusion_matrix[i, j],
    #                        ha="center", va="center", color="black", weight="heavy")
    #
    # # 添加颜色条图例
    # # cbar = ax.figure.colorbar(im, ax=ax)
    #
    # # 将图像保存为 JPG 图像
    # plt.savefig(metrixpath, dpi=300)
    labels = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']

    # Compute the percentages of each class
    class_counts = np.sum(confusion_matrix, axis=1)
    percentages = confusion_matrix / class_counts[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the confusion matrix with color scaling based on percentages
    im = ax.imshow(percentages, cmap='Blues', vmin=0, vmax=1)
    # 'Reds', 'Greens', 'Oranges', and 'viridis'

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=16)
    ax.set_yticklabels(labels, fontsize=16)
    ax.set_xlabel('True class', fontsize=20)
    ax.set_ylabel('Predicted class', fontsize=20)

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = confusion_matrix[i, j]
            text_color = 'white' if value == np.max(confusion_matrix[:, j]) else 'black'
            text = ax.text(j, i, value, ha="center", va="center", color=text_color, fontweight='bold', fontsize=14)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.savefig(metrixpath, dpi=1200)


if __name__ == "__main__":

    basepath = "../weight/lunwen/convnext_no_0.01_100_224 86.56/convnext_0.01_100_224.pth"
    # savepath = "../csv/lunwen/summaryconvnext_SCECAjiangzaoaff0.005_100.csv"
    savepath = "../weight/lunwen/convnext_no_0.01_100_224 86.56/summaryconvnext_0.01_100_224.csv"
    metrixpath = 'C:\\zlg\\classification_isic19\\weight\\convnext_SCECAjiangzaoaff0.005_100.jpg'

    summary(savepath)
    name, acc, matrix = get_matrix()
    draw(matrix, metrixpath)