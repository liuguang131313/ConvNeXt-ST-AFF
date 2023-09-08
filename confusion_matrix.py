"""
混淆矩阵
@date: 2022/05/01
@author: wuqiwei
"""
import numpy
from prettytable import PrettyTable


class ConfusionMatrix(object):

    def __init__(self, class_num: int):
        self.matrix = numpy.zeros((class_num, class_num))
        self.class_num = class_num

    def update(self, pred, label):
        # p代表Predicted label、t代表True label
        for p, t in zip(pred, label):
            self.matrix[p, t] += 1

    def acc(self):
        acc = 0
        for i in range(self.class_num):
            acc += self.matrix[i, i]
        acc = acc / numpy.sum(self.matrix)
        return acc


