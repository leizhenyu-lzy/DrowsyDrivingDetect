# 1950083 自动化 刘智宇
import torch
import cv2 as cv
import numpy as np
import pandas as pd
import os.path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 记录数据集路径
nets_save_dir = r"D:\Project\DrowsyDrivingDetect\KeyPointsDetect\Models"
dataset_images_base_dir = r"D:\Project\DataSet\WFLW\WFLW_images"

train_dataset_annotation_dir = r'D:\Project\DataSet\WFLW\WFLW_annotations\list_98pt_rect_attr_train_test\list_98pt_rect_attr_train.txt'
test_dataset_annotation_dir = r"D:\Project\DataSet\WFLW\WFLW_annotations\list_98pt_rect_attr_train_test\list_98pt_rect_attr_test.txt"

# Net Choices
NetNames = ["KPNet", "WithAlexNet", "WithResNet18", "WithResNet34", "WithResNet50", "WithGoogleNet", "WithVgg16"]
kpnet_idx = 0
with_alexnet_idx = kpnet_idx + 1
with_resnet18_idx = with_alexnet_idx + 1
with_resnet34_idx = with_resnet18_idx + 1
with_resnet50_idx = with_resnet34_idx + 1
with_resnet101_idx = with_resnet50_idx + 1
with_googlenet_idx = with_resnet101_idx + 1
with_vgg16_idx = with_googlenet_idx + 1


# LossFunc Choices
LossFuncNames = ["MSELoss", "MAELoss", "CrossEntropyLoss", "HuberLoss"]
mse_idx = 0
mae_idx = mse_idx + 1
cross_entropy_idx = mae_idx + 1
huber_idx = cross_entropy_idx + 1


# Optimizer
OptimNames = ["SGDOptim", "AdamOptim", "AdamaxOptim"]
sgd_idx = 0
adam_idx = sgd_idx + 1


# DataLoader Consts
train_shuffle = True;   test_shuffle = True
train_drop_last = True; test_drop_last = True
train_num_workers = 0;  test_num_workers = 0


# 可改动的地方
train_epoch = 2
train_batch_size = 5;   test_batch_size = 5
train_lr = 0.01
test_show_train_result_steps = 250  # 训练到多少次的时候输出结果

net_choice = with_resnet34_idx  # 选择模型类型
optim_choice = sgd_idx
lossfunc_choice = mae_idx


# 模型列表(需要和上面的net_choice进行匹配，否则会不能成功导入训练好的模型)
"""
按训练顺序排列：
1. Models/GPU_WithResNet18_SGDOptim_MAELoss_Epoch10_BatchSize50_LR0.01_MaxBatchLoss11.956532.pth
2. Models/GPU_WithResNet18_SGDOptim_MAELoss_Epoch6_BatchSize50_LR0.01_MaxBatchLoss16.683079.pth
3. Models/GPU_WithResNet34_SGDOptim_MAELoss_Epoch6_BatchSize5_LR0.01_MaxBatchLoss23.802553.pth
"""
use_model_name = r"Models/GPU_WithResNet34_SGDOptim_MAELoss_Epoch6_BatchSize5_LR0.01_MaxBatchLoss23.802553.pth"


# 文件名称
network_name_without_suffix = NetNames[net_choice] + '_' + OptimNames[optim_choice] + \
                        '_' + LossFuncNames[lossfunc_choice] + "_Epoch" + str(train_epoch) + \
                        "_BatchSize" + str(train_batch_size) + "_LR"+str(train_lr)

# unify <=> face_cut + resize
# 图片resize后图片大小
unify_image_size = [224, 224]

"""
关键点序号
我们关心的关键点，具体可查看：WFLW_annotation.png
眼睛轮廓：60-67 68-75
眼睛中心：96 97
嘴唇外轮廓：76-87
嘴唇内轮廓：88-95
从0开始
原标签：2*98+4+6+1=207
现标签：2*(8+8+12+8)+4+6-6+1=83-6=77（有6个标签暂未使用）
"""
# 下面这些是点的序号，具体取数值(x_pos,y_pos)需要（*2，*2+1）
# 开头有下划线的是origin_annotation，没有的是simple_annotation的
_left_eye_start = 60;           left_eye_start = 0
_left_eye_end = 67;             left_eye_end = 7
_right_eye_start = 68;          right_eye_start = 8
_right_eye_end = 75;            right_eye_end = 15
_outer_lip_start = 76;          outer_lip_start = 16
_outer_lip_end = 87;            outer_lip_end = 27
_inner_lip_start = 88;          inner_lip_start = 28
_inner_lip_end = 95;            inner_lip_end = 35
key_points_numbers = 36  # 8(眼)+8(眼)+12(外唇)+8(内唇)
# 下面这些是annotation的index（原有标注完整标注的index），直接调用即可
_x_min_rect_idx = 196;          x_min_rect_idx = 72;            col_min_rect_idx = x_min_rect_idx
_y_min_rect_idx = 197;          y_min_rect_idx = 73;            row_min_rect_idx = y_min_rect_idx
_x_max_rect_idx = 198;          x_max_rect_idx = 74;            col_max_rect_idx = x_max_rect_idx
_y_max_rect_idx = 199;          y_max_rect_idx = 75;            row_max_rect_idx = y_max_rect_idx
_img_relative_root_idx = 206;   img_relative_root_idx = 76
# 暂时没用到
# _pose_idx = 200;            pose_idx = 76
# _expression_idx = 201;      expression_idx = 77
# _illumination_idx = 202;    illumination_idx = 78
# _make_up_idx = 203;         make_up_idx = 79
# _occlusion_idx = 204;       occlusion_idx = 80
# _blur_idx = 205;            blur_idx = 81

# 分割线
separate_bar = "----------"
train_dataset_name = "WFLW_train"
test_dataset_name = "WFLW_test"


# Color
bgr_Blue = (255, 0, 0)
bgr_Green = (0, 255, 0)
bgr_Red = (0, 0, 255)
bgr_Black = (0, 0, 0)
bgr_White = (255, 255, 255)
bgr_Yellow = (0, 255, 255)
bgr_Pink = (204, 51, 255)
bgr_Purple = (153, 51, 153)



"""
name:       read_annotation
functional: use pandas to read the annotation.txt and get the simplified information
inputs:     root_dir : directory of the annotation file (default : train_dataset_annotation_dir)
outputs:    annotation  (<class 'numpy.ndarray'>)
            size of the origin annotation 
            size of the simple annotation
"""
def read_annotation(root_dir=train_dataset_annotation_dir):
    # 使用pandas读取完整的标注进入DataFrame
    origin_annotation = pd.read_csv(filepath_or_buffer=root_dir, sep=' ', header=None, index_col=None)
    origin_annotation = origin_annotation.values  # 转化为列表形式
    origin_annotation_shape = origin_annotation.shape
    # print(origin_annotation_shape)
    # 取出关注的关键点，简化标注
    simple_annotation_p1 = origin_annotation[:, _left_eye_start*2:_inner_lip_end*2+2]  # 眼睛嘴唇的关键点正好连续
    simple_annotation_p2 = origin_annotation[:, _x_min_rect_idx:_y_max_rect_idx+1]  # 脸部矩形的两个点
    simple_annotation_p3 = origin_annotation[:, _img_relative_root_idx]  # 对应图像路径
    simple_annotation_p3 = simple_annotation_p3[:, np.newaxis]  # 添加新维度，否则无法进行concatenate操作
    # print(simple_annotation_p1.shape, type(simple_annotation_p1))  # (7500, 72) <class 'numpy.ndarray'>
    # print(simple_annotation_p2.shape, type(simple_annotation_p2))  # (7500, 4) <class 'numpy.ndarray'>
    # print(simple_annotation_p3.shape, type(simple_annotation_p3))  # (7500, 1) <class 'numpy.ndarray'>
    # 合成简化的标注
    simple_annotation = np.concatenate((simple_annotation_p1, simple_annotation_p2, simple_annotation_p3), axis=1)
    # print(simple_annotation)
    simple_annotation_shape = simple_annotation.shape
    # print(simple_annotation_shape)  # (7500, 77)
    return simple_annotation, origin_annotation_shape, simple_annotation_shape


if __name__ == "__main__":
    print(network_name_without_suffix)  # Withresnet34_Epoch10_BatchSize150_LR0.01.pth
    print(os.path.isfile(network_name_without_suffix))  # False(但其实也能打开)
    print(network_name_without_suffix)  # D:\Project\DrowsyDrivingDetect\KeyPointsDetect\Models\KPNet_SGDOptim_MSELoss_Epoch1_BatchSize50_LR0.05.pth
    print(os.path.isabs(network_name_without_suffix))  # True
    print(os.path.isfile(network_name_without_suffix))  # False




