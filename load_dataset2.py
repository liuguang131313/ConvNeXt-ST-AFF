"""
加载数据集
@date: 2022/05/01
@author: wuqiwei
"""
import argparse
import os
from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
# from torch.utils.data.sampler import WeightedRandomSampler
# from torch.utils.data.sampler import *

# from imblearn.over_sampling import RandomOverSampler

from config import *
# from utils.sample import *

import numpy as np
import pandas as pd

def str2bool(v):
    """
    Converts string to bool type; enables command line
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script for image classification', add_help=False)
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    parser.add_argument('--nb_classes', default=class_num, type=int,
                        help='number of the classification types')
    parser.add_argument('--input_size', default=img_s, type=int,
                        help='image input size')
    parser.add_argument('--train_path', default=train, type=str,
                        help='dataset path')
    parser.add_argument('--valid_path', default=valid, type=str,
                        help='dataset path for evaluation')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', type=str2bool, default=False,
                        help='Do not random erase first (clean) augmentation split')
    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    return parser

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
    # mean = dataset_mean
    # std = dataset_std

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:
            t.append(
            transforms.Resize((args.input_size, args.input_size),
                            interpolation=transforms.InterpolationMode.BICUBIC),
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def get_data_loader(args):
    """
    ImageFolder和DataLoader加载数据集
    :return: train_loader, valid_loader
    """
    transform_train = build_transform(is_train=True, args=args)
    print("transform_train = ")
    for t in transform_train.transforms:
        print(t)
    print("---------------------------")
    transform_valid = build_transform(is_train=False, args=args)
    print("transform_train = ")
    for t in transform_valid.transforms:
        print(t)
    print("---------------------------")

    train_dataset = ImageFolder(args.train_path, transform=transform_train)
    valid_dataset = ImageFolder(args.valid_path, transform=transform_valid)
    nb_classes = args.nb_classes
    assert len(train_dataset.class_to_idx) == nb_classes
    print("Number of the class = %d" % nb_classes)

    # class_names = train_dataset.classes
    # n_class = len(class_names)
    # 映射关系：类别 到 索引号
    train_dataset.class_to_idx
    # 映射关系：索引号 到 类别
    idx_to_labels = {y: x for x, y in train_dataset.class_to_idx.items()}
    # 保存为本地的 npy 文件
    np.save('idx_to_labels.npy', idx_to_labels)
    np.save('labels_to_idx.npy', train_dataset.class_to_idx)

    target = train_dataset.targets
    target1 = valid_dataset.targets
    class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
    class_sample_count1 = np.array([len(np.where(target1 == t)[0]) for t in np.unique(target1)])
    # for i in class_sample_count:
    #     weights = len(train_dataset) / i
    #     print(weights)
    #     weight.append(weights)
    weight = len(train_dataset) / (class_sample_count * len(train_dataset.classes))
    samples_weight = torch.from_numpy(weight)
    # print(samples_weight)
    # samples_weight = np.array([samples_weight[t] for t in target])
    # print(samples_weight)
    # samples_weight = torch.from_numpy(samples_weight)
    # samples_weight = samples_weight.double()

    print('训练集图像数量', len(train_dataset))
    print('类别个数', len(train_dataset.classes))
    print('各类别名称', train_dataset.classes)
    print("各类被数量", class_sample_count)
    print("各类别权重", samples_weight)

    print('测试集图像数量', len(valid_dataset))
    print('类别个数', len(valid_dataset.classes))
    print('各类别名称', valid_dataset.classes)
    print("各类被数量", class_sample_count1)

    # sampler_train = torch.utils.data.DistributedSampler(
    #     dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
    # )

    # # 计算各类别的样本数量
    # class_counts = [0] * len(train_dataset.classes)
    # for _, label in train_dataset:
    #     class_counts[label] += 1

    # 计算各类别的权重
    # total_samples = sum(class_counts)
    # class_weights = [total_samples / (len(train_dataset.classes) * count) for count in class_counts]
    # print(class_weights)

    # print("Class weights:", class_weights)

    # trainsampler = WeightedRandomSampler(samples_weight, len(train_dataset))
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              # sampler=trainsampler,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=num_workers,
                              drop_last=True
    )

    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=num_workers,
                              drop_last=False
                              )

    # 输出每个batch的图像标签
    # for i, (images, labels) in enumerate(train_loader):
    #     print('Batch', i + 1, 'labels:', labels)
    # num_training_steps_per_epoch = len(train_dataset) // batch_size

    return train_loader, valid_loader, samples_weight

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    get_data_loader(args)
