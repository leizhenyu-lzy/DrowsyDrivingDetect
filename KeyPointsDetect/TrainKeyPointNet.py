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
import GraphicDisplay


if __name__ == "__main__":
    # ---------------------- 前期准备工作工作 ----------------------
    print(separate_bar * 2, "模型训练准备工作开始", separate_bar * 2)
    # 对数据集、标注信息进行检查
    print("\n" + "WFLW数据集路径检查开始：")
    Validation = ToolFunction.check_path()
    if not Validation:
        print("The dataset path is wrong.")
        exit(1)

    # 对原有标注进行简化（只取出关键列），并获取原标注信息和简化标注信息
    print("\n" + "读取数据集annotation：")
    train_simple_annotation, train_origin_annotation_shape, train_simple_annotation_shape = read_annotation(root_dir=train_dataset_annotation_dir)
    test_simple_annotation, test_origin_annotation_shape, test_simple_annotation_shape = read_annotation(root_dir=test_dataset_annotation_dir)

    # 数据集相关信息展示
    print("\n" + "数据集相关信息：")
    ToolFunction.inform_dataset_basic(train_origin_annotation_shape, train_simple_annotation_shape)
    ToolFunction.inform_dataset_basic(test_origin_annotation_shape, test_simple_annotation_shape)
    num_files = ToolFunction.count_files(dataset_images_base_dir)
    print("\n" + "数据集图像数量： ", num_files)

    # 测试GPU是否可用
    print("\n" + "测试GPU是否可以使用：")
    useGPU = ToolFunction.testGPU()
    if useGPU:
        network_name_prefix = "GPU_"
    else:
        network_name_prefix = "CPU_"

    # 对simple_annotation进行处理
    print("\n" + "拆分数据集annotation：")
    train_simple_coords_annotation_float = train_simple_annotation[:, 0:img_relative_root_idx]  # 只取出坐标标注
    train_simple_coords_annotation_int = train_simple_coords_annotation_float.astype(np.int32)  # 讲坐标标注转为整形方便opencv处理
    train_img_paths_annotation = train_simple_annotation[:, -1]  # 只取出文件路径标注

    test_simple_coords_annotation_float = test_simple_annotation[:, 0:img_relative_root_idx]  # 只取出坐标标注
    test_simple_coords_annotation_int = test_simple_coords_annotation_float.astype(np.int32)  # 讲坐标标注转为整形方便opencv处理
    test_img_paths_annotation = test_simple_annotation[:, -1]  # 只取出文件路径标注

    # 获取unify_coords
    print("\n" + "unify数据集annotation：")
    print("训练集：")
    train_unify_coords = ToolFunction.unify_img_coords_annotation(train_simple_coords_annotation_int, train_img_paths_annotation)
    print("测试集：")
    test_unify_coords = ToolFunction.unify_img_coords_annotation(test_simple_coords_annotation_int, test_img_paths_annotation)
    # unify_coords和simple_coords_annotation_int大小一致，只是经过同一化。矩形框位置保留。

    # 查找数据集unify后关键点坐标存在小于零的图片（即在原图中关键点不在人脸矩形框的图片）  # img_idx:  1035  # img_idx:  2747
    # print("\n" + "检查数据集中有错误的数据")
    # ToolFunction.show_key_points_in_incomplete_unify_img(unify_coords, simple_coords_annotation_int, img_paths_annotation)

    print(separate_bar * 2, "模型训练准备工作结束", separate_bar * 2)
    # 模型训练准备工作

    # ---------------------- 模型训练 ----------------------
    print(separate_bar * 2, "模型训练开始", separate_bar * 2)
    net = KeyPointNet(net_choice);      net = net.to(device)  # 转入GPU 其实可以不用重新赋值
    train_dataset = WFLW_Dataset(unify_coords_anno=train_unify_coords, img_paths_anno=train_img_paths_annotation)
    test_dataset = WFLW_Dataset(unify_coords_anno=test_unify_coords, img_paths_anno=test_img_paths_annotation)

    # 返回 : ① <class 'torch.Tensor'> torch.float32  ② <class 'numpy.ndarray'> int32
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=train_shuffle,
                                  num_workers=train_num_workers, drop_last=train_drop_last)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=test_shuffle,
                                 num_workers=test_num_workers, drop_last=test_drop_last)

    Loss = GetLoss();   Loss = Loss.to(device)  # 转入GPU
    Optimizer = GetOptimizer(optim_idx=sgd_idx, parameter=net.parameters(), LR=train_lr)

    train_loss = torch.tensor(0.0, dtype=torch.float32);        train_loss = train_loss.to(device)
    test_loss = torch.tensor(0.0, dtype=torch.float32);         test_loss = train_loss.to(device)
    test_average_loss = torch.tensor(0.0, dtype=torch.float32); test_average_loss = train_loss.to(device)
    test_sum_loss = torch.tensor(0.0, dtype=torch.float32);     test_sum_loss = train_loss.to(device)
    epoch_loss = torch.tensor(0.0, dtype=torch.float32);        epoch_loss = epoch_loss.to(device)
    max_batch_loss = torch.tensor(0.0, dtype=torch.float32);    max_batch_loss = max_batch_loss.to(device)  # 记录最后一个epoch的最大batchloss
    total_train_steps = 0

    start_time = time.time()  # 计时起点

    train_loss_list = [];   test_loss_list = []
    batch_step_list = [];   epoch_step_list = []

    # 训练
    net.train()
    for epoch_index in range(train_epoch):
        print("Epoch: {}".format(epoch_index), " Start.")
        # 归零
        epoch_loss = 0.0
        max_batch_loss = 0.0
        for batch_index, inputs in enumerate(train_dataloader):
            loss = 0.0
            unify_gray_img, actual_coords = inputs  # 变量必须inplace
            unify_gray_img = unify_gray_img.to(device)
            actual_coords = actual_coords.to(device)
            actual_coords = actual_coords.float()  # 不进行数据类型转换无法进行loss.backward()

            train_coords = net(unify_gray_img)  # 训练出的网络给出train_coords
            # print(unify_gray_img.shape)  # torch.Size([train_batch_size, 1, 224, 224])
            # print(actual_coords.dtype, train_coords.dtype, loss.dtype)  # torch.float32  torch.float32  torch.float32
            # print(actual_coords.shape, train_coords.shape)  # torch.Size([train_batch_size, 72]) torch.Size([train_batch_size, 72])
            train_loss = Loss(actual_coords, train_coords)  # 两个都是 <class 'torch.Tensor'> 数据类型
            # print(actual_coords.dtype, train_coords.dtype, loss.dtype)  # torch.float32 torch.float32 torch.float32
            # print(train_loss)  # tensor(float_number, grad_fn=<MseLossBackward0>)

            Optimizer.zero_grad()  # 将之前计算的梯度清空
            train_loss.backward()
            Optimizer.step()

            # 训练到一定步骤后进行结果展示
            if total_train_steps % train_result_show_steps == 0:
                end_time = time.time()
                print("Epoch:{}  Batch:{}  TrainSteps:{}  BatchLoss:{}  BatchTimeConsume:{}".format(epoch_index, batch_index, total_train_steps, loss, end_time-start_time))

            # 记录最后一个epoch的最大batch_loss
            if loss > max_batch_loss:
                max_batch_loss = loss

            # 将数据添加进列表中
            train_loss_list.append(train_loss.detach().cpu())
            batch_step_list.append(total_train_steps)

            total_train_steps = total_train_steps + 1   # 训练次数数递增
            epoch_loss += loss  # 每一个batch的训练loss累计到epoch的loss

        # 训练完一个epoch，在测试集上进行测试
        with torch.no_grad():
            net.eval()
            for test_inputs in test_dataloader:
                test_unify_gray_img, test_actual_coords = test_inputs
                test_unify_gray_img = test_unify_gray_img.to(device)
                test_actual_coords = test_actual_coords.to(device)
                test_actual_coords = test_actual_coords.float()
                test_train_coords = net(test_unify_gray_img)
                test_loss = Loss(test_actual_coords, test_train_coords)
                test_sum_loss += test_loss

            test_average_loss = test_sum_loss / (test_simple_annotation_shape[0])
        test_loss_list.append(test_average_loss)
        epoch_step_list.append(total_train_steps - 1)  # -1是为了使得epoch结束的位置统一（每个batch结束的时候steps会递增）

        # 输出一个Epoch的结果
        end_time = time.time()  # 计时终点
        print("Epoch:{}  End.  EpochLoss:{}  MaxBatchLoss:{}  EpochTimeConsume:{}".format(epoch_index, epoch_loss, max_batch_loss, end_time-start_time))
    print(separate_bar * 2, "模型训练结束", separate_bar * 2)
    # 模型训练结束

    # ---------------------- 网络训练收尾工作 ----------------------
    print(separate_bar * 2, "网络训练收尾工作开始", separate_bar * 2)

    # 模型保存开始
    print("\n" + "网络保存：")
    max_batch_loss = str(max_batch_loss.detach().cpu().numpy())
    network_name_without_suffix = network_name_prefix + network_name_without_suffix
    network_save_path_without_suffix = os.path.join(nets_save_dir, network_name_without_suffix)
    network_complete_save_path = network_save_path_without_suffix + "_MaxBatchLoss" + str(max_batch_loss) + ".pth"
    # torch.save(net, network_complete_save_path)
    print(network_complete_save_path)
    torch.save(net.state_dict(), network_complete_save_path)
    # print(separate_bar * 2, "模型训练收尾工作结束", separate_bar * 2)
    # print(separate_bar * 2, "模型训练收尾工作结束", separate_bar * 2)

    # ---------------------- 网络模型可视化展示和训练结果保存 ----------------------
    print("\n" + "训练过程可视化：")
    GraphicDisplay.GraphicDisplayLoss(net_name=network_complete_save_path, train_loss_list=train_loss_list,
                                      test_loss_list=test_loss_list, batch_step_list=batch_step_list, epoch_step_list=epoch_step_list)
