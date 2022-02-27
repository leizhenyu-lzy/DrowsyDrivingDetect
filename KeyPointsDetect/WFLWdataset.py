# 1950083 自动化 刘智宇
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2 as cv
import numpy as np
import pandas as pd
import os
from Consts import *
import ToolFunction

"""
name:       WFLW_Dataset
functional: 继承Dataset类
inputs:     unify_coords_annotation : 
            img_path_annotation : 图片相对地址
            whether_unify_img : 传入的图片是否经过unify(default:False)
            dataset_root_dir : 数据集图片根目录(default:dataset_images_base_dir)
outputs:    None
"""
class WFLW_Dataset(Dataset):
    def __init__(self, unify_coords_anno, img_paths_anno, whether_unify_img=False, dataset_root_dir=dataset_images_base_dir):
        self.unify_coords_annotation = unify_coords_anno
        self.img_paths_annotation = img_paths_anno
        self.len = len(img_paths_anno)
        self.whether_unify_img = whether_unify_img
        self.dataset_root_dir = dataset_root_dir

    def __getitem__(self, idx):
        unify_coords = self.unify_coords_annotation[idx]  # 取出该idx的图片
        unify_labels = unify_coords[:-4]
        img_path = self.img_paths_annotation[idx]
        # 如果不是绝对地址就将其转换为绝对地址
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.dataset_root_dir, img_path)
        # 不是unify图片，要进行裁剪处理，路径要加上前缀变为绝对路径
        if not self.whether_unify_img:
            img_path = os.path.join(dataset_images_base_dir, self.img_paths_annotation[idx])
            img_gray_whole = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            img_gray_rect_cut = img_gray_whole[unify_coords[row_min_rect_idx]:unify_coords[row_max_rect_idx],
                                               unify_coords[col_min_rect_idx]:unify_coords[col_max_rect_idx]]
            img_gray_unify = cv.resize(img_gray_rect_cut, unify_gray_image_size)
            # cv.imshow("img_gray_unify", img_gray_unify)
            # print(img_gray_unify.shape)
            # cv.waitKey(1000)
            # cv.destroyAllWindows()
        # 是unify图片，已经经过裁剪和resize，直接读取即可
        else:  # *********还未验证*********
            img_path = os.path.join(dataset_images_base_dir, self.img_paths_annotation[idx])
            img_gray_unify = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

        # print(type(img_gray_unify))  # <class 'numpy.ndarray'>  # 读取图片的数据类型为ndarray   # 进行了数据类型转换
        img2tensor_tf = transforms.ToTensor()
        tensor_unify = img2tensor_tf(pic=img_gray_unify)
        # totensor()Image或numpy.ndarray，将其先由HWC转置为CHW格式，再转为float后每个像素除以255

        # 返回值数据类型说明
        # tensor_unify  <class 'torch.Tensor'>  torch.Size([1, 224, 224])   torch.float32
        # unify_labels  <class 'numpy.ndarray'> (72,)                       int32
        return tensor_unify, unify_labels

    def __len__(self):
        return self.len


if __name__ == "__main__":
    pass
