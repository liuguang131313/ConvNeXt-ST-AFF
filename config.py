"""
参数配置
数据集：痤疮-a，黄褐斑-b，玫瑰痤疮-c，太田痣-d
@date: 2022/05/01
@author: wuqiwei
"""
# gpu参数
is_cuda = True

device = 'cuda:0'

is_parallel = False
gpu_id = "0"
gpu_ids = [0]
# 数据集参数

#ISIC
# dataset_mean = (0.6679, 0.5293, 0.5238)
# dataset_std = (0.1880, 0.1902, 0.2016)
# dataset_mean = (0.6677, 0.5297, 0.5244)
# dataset_std = (0.1762, 0.1813, 0.1929)

# 痤疮


#子宫
dataset_mean = (0.5613, 0.5778, 0.6032)
dataset_std = (0.2114, 0.1957, 0.1590)


# 预训练权重
base_weight1 = "weight/base/InceptionV3.pth"
base_weight2 = "weight/base/ResNet50.pth"
base_weight3 = "weight/base/swin_tiny_patch4_window7_224.pth"
base_weight4 = "weight/base/adv-efficientnet-b0-b64d5a18.pth"

# base_weight5 = "weight/base/convnext_base_1k_224_ema.pth"
base_weight5 = "weight/base/convnext_base_22k_224.pth"
# base_weight5 = "weight/base/swin_base_patch4_window7_224_22k.pth"
base_weight6 = "weight/base/swin_base_patch4_window7_224_22k.pth"
# base_weight6 = "weight/base/biformer_base_best.pth"
# 训练相关超参数
# net_name = "efficientnet"
# net_name = "effi"
# net_name = "resnet"
# net_name = "swintransformer"
net_name = "convnext"

# class_num = 8
batch_size = 8
num_workers = 0
# max_epoch = 60
warmup_epochs = 5
warmup_steps = -1

# lr = 0.01
# lr = 0.0005
min_lr = 1e-6
weight_decay = 0.05
# weight_decay = 1e-4

# milestones = [25, 50, 75]
# gamma = 0.1
milestones = [20, 40, 60, 80]
gamma = 0.5

#mixuo数据增强
# mixup = 0.8  # 0.8
# cutmix = 1.0  # 1.0
# # mixup = 0  # 0.8
# # cutmix = 0  # 1.0
# mixup_prob = 1.0
# mixup_switch_prob = 0.5
# mixup_mode ='batch'
# smoothing = 0.1
# cutmix_minmax = None

img_s = 224
# img_s = 384
# img_s = [224, 224]

# class_num = 8
# train = "C:\\zlg\\ConvNeXt\\ISIC2019_split\\train"
# valid = "C:\\zlg\\ConvNeXt\\ISIC2019_split\\val"


# lr = 0.0005
# max_epoch = 100
# class_num = 8
# train = "C:\\zlg\\ISIC2019\\train"
# valid = "C:\\zlg\\ISIC2019\\valid"

lr = 0.001
max_epoch = 60
class_num = 6
train = "C:\\zlg\\dataset_split\\train"
valid = "C:\\zlg\\dataset_split\\valid"

# class_num = 2
# train= "C:\\zlg\\cell2\\train"
# valid= "C:\\zlg\\cell2\\val"

#调整
drop_path = 0.8  # 0.8， 0.1 ，0.2