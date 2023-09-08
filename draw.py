"""
绘图
@date: 2022/05/01
@author: wqw
"""
import pandas
# import matplotlib.pyplot as plt
import torch

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"   # 表示允许重复加载动态链接库

def draw():
    # csv = pandas.read_csv("csv/convnextsc0.01.csv")
    csv = pandas.read_csv("csv/convnextjiangzao0.01.csv")
    x = csv.values[:, 0]
    y1 = csv.values[:, 1]
    y2 = csv.values[:, 2]
    y3 = csv.values[:, 3]
    y4 = csv.values[:, 4]
    draw_show(x, y1, y2, y3, y4)


def draw_show(x, y1, y2, y3, y4):
    plt.figure()
    # plt.plot(x, y1, label="train_loss", linestyle="-", color='#8A2BE2')
    plt.plot(x, y2, label="train_acc", linestyle="-", color='#77933C')
    # plt.plot(x, y3, label="valid_loss", linestyle="-", color='#558ED5')
    plt.plot(x, y4, label="valid_acc", linestyle="-", color='#E46C0A')
    plt.legend()
    plt.title("Valid Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    # 保存图片
    plt.savefig('C:\\zlg\\classification_isic19\\weight\\convnext_jiangzao0.01.jpg', dpi=300)
    plt.show()

def get_matrix():
    maps = torch.load("weight/save/convnext_jiangzao0.01.pth")
    print(maps["epoch"])
    print(maps["best_acc"])
    print(maps["best_matrix"])
    return maps["best_matrix"]


if __name__ == "__main__":
    draw()
    # get_matrix()
