# 1950083 自动化 刘智宇
import torch
import torchvision
from torch.utils.data import DataLoader

import os
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from Consts import *
import ToolFunction
from WFLWdataset import WFLW_Dataset
from KeyPointNet import *

if __name__ == "__main__":
    # # 验证
    # simple_annotation, origin_annotation_size, simple_annotation_size = read_annotation()
    # simple_coords_annotation_float = simple_annotation[:, 0:img_relative_root_idx]  # 只取出坐标标注
    # simple_coords_annotation_int = simple_coords_annotation_float.astype(np.int32)  # 讲坐标标注转为整形方便opencv处理
    # img_paths_annotation = simple_annotation[:, -1]  # 只取出文件路径标注
    # unify_coords = ToolFunction.unify_img_coords_annotation(simple_coords_annotation_int, img_paths_annotation)
    # train_dataset = WFLW_Dataset(unify_coords_anno=unify_coords, img_paths_anno=img_paths_annotation)
    #
    # net = KeyPointNet(NetChoice=net_choice)
    #
    # img_idx = 1000
    # ToolFunction.show_train_key_points_in_unify_img(net, train_dataset, img_paths_annotation, simple_coords_annotation_int, img_idx, 0)
    #
    #
    # # 使用show_key_points_in_unify_img
    # """
    # img_idx = 345
    # img_path = os.path.join(dataset_images_base_dir, img_paths_annotation[img_idx])
    # ToolFunction.show_key_points_in_unify_img(unify_coords[img_idx], img_paths_annotation[img_idx], wait_time=0)
    # print(type(unify_coords[img_idx]), unify_coords[img_idx].shape)
    # print(img_paths_annotation[img_idx])
    # """
    # """
    # x = torch.ones((1, 72))
    # x = x.numpy()
    # print(type(x), x.shape, x.dtype)  # <class 'numpy.ndarray'> (1, 72) float32
    # x = np.reshape(x, -1)  # 拉平
    # print(x.shape)
    # test = [1,2,3,4,5,6]
    # x = np.append(x, test[0:4])
    # print(x, x.shape)
    # """
    simple_annotation, origin_annotation_size, simple_annotation_size = read_annotation()
    simple_coords_annotation_float = simple_annotation[:, 0:img_relative_root_idx]  # 只取出坐标标注
    simple_coords_annotation_int = simple_coords_annotation_float.astype(np.int32)  # 讲坐标标注转为整形方便opencv处理
    img_paths_annotation = simple_annotation[:, -1]  # 只取出文件路径标注
    unify_coords = ToolFunction.unify_img_coords_annotation(simple_coords_annotation_int, img_paths_annotation)
    train_dataset = WFLW_Dataset(unify_coords_anno=unify_coords, img_paths_anno=img_paths_annotation)

    img_idx = 2747
    ToolFunction.show_key_points_and_rect_in_origin_img(simple_coords_annotation_int[img_idx], img_paths_annotation[img_idx], wait_time=0)


