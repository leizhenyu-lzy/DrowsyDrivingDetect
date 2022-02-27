# 1950083 自动化 刘智宇
import torch
import os
import cv2 as cv
import numpy as np
import pandas as pd
from Consts import *


print()


"""
name:       count_files
functional: count number of files in a directory(use:os.walk)
inputs:     root_dir
outputs:    number of files in the directory
"""
def count_files(root_dir):
    num_files = 0
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            # print(os.path.join(root, file_name))
            num_files = num_files + 1
    return num_files


"""
name:       unify_img_coords_simple_annotation
functional: unify the coords of the image(s)
inputs:     not_unify_coords : 简化过但是没有同一化的coords annotation
            img_dir : 图片路径。如果是单张图片，使用时也需要放入列表中
outputs:    unified coords
"""
def unify_img_coords_annotation(not_unify_coords, img_paths):
    # print(separate_bar, "进行坐标unify", separate_bar)
    # 判断是单张图片还是多张图片(list)
    num_of_img = len(img_paths)
    if num_of_img == 1:  # 如果是单张图片增加一个新维度，提高代码复用
        print("Single Image.")
        not_unify_coords = not_unify_coords[np.newaxis, :]
    else:
        print("Multiple Images.")
    print("共有 {} 张图片需要处理".format(num_of_img))

    # 创建一个大小相同的矩阵
    unify_coords = not_unify_coords.copy()
    # print(unify_coords.dtype)  # int32
    # unify同一化
    for img_idx in range(num_of_img):
        x_min = not_unify_coords[img_idx][x_min_rect_idx]; x_max = not_unify_coords[img_idx][x_max_rect_idx]
        y_min = not_unify_coords[img_idx][y_min_rect_idx]; y_max = not_unify_coords[img_idx][y_max_rect_idx]
        rect_x = x_max - x_min; rect_y = y_max - y_min
        for point_idx in range(left_eye_start, inner_lip_end * 2 + 2, 2):
            unify_coords[img_idx][point_idx] = (not_unify_coords[img_idx][point_idx]-x_min) * unify_gray_image_size[0] / rect_x
        for point_idx in range(left_eye_start + 1, inner_lip_end * 2 + 2, 2):
            unify_coords[img_idx][point_idx] = (not_unify_coords[img_idx][point_idx]-y_min) * unify_gray_image_size[1] / rect_y

    # 输出验证信息，返回结果
    # print(unify_coords, unify_coords.shape, unify_coords.dtype)  # (7500, 76) int32
    # print(not_unify_coords, not_unify_coords.shape)  # (7500, 76)
    # print(separate_bar, "坐标unify完成", separate_bar)
    return unify_coords


"""
name:       show_key_points_and_rect_in_origin_img
functional: use the origin image and unified coords to point the key points
inputs:     unify_coords : 一定是简化的坐标
            img_path : 图片路径
            imread_type : cv.imread("img_dir", flags)中flags参数，仅用于传递参数(default:cv.IMREAD_COLOR)
            wait_time : cv.waitKey(wait_time)时间参数，仅用于传递参数(default:1000(1秒))
outputs:    None
"""
def show_key_points_and_rect_in_origin_img(not_unify_coords, img_path, imread_type=cv.IMREAD_COLOR, wait_time=0):
    # 如果是不完整的路径(仅有上一级文件夹和图片名)则将其补全(加上文件夹所在位置)
    if not os.path.isabs(img_path):
        img_path = os.path.join(dataset_images_base_dir, img_path)

    img_origin = cv.imread(img_path, flags=imread_type)
    x_min = not_unify_coords[x_min_rect_idx];   x_max = not_unify_coords[x_max_rect_idx]
    y_min = not_unify_coords[y_min_rect_idx];   y_max = not_unify_coords[y_max_rect_idx]
    cv.rectangle(img_origin, pt1=(x_min, y_min), pt2=(x_max, y_max), color=bgr_Green, thickness=1)
    for i in range(key_points_numbers):
        cv.circle(img_origin, center=(not_unify_coords[2 * i], not_unify_coords[2 * i + 1]), radius=0, color=bgr_Pink, thickness=3)
    cv.imshow("img_origin", img_origin)
    cv.waitKey(wait_time)
    cv.destroyAllWindows()


"""
name:       show_key_points_in_unify_img
functional: use the origin image (or unify image) and unified coords to point the unified key points
inputs:     unify_coords : 一定是简化的坐标(可以接受的参数类型: <class 'numpy.ndarray'> (76,))
            img_path : 图片路径（可以是完整路径，也可以是不完整的路径（即只有图片文件夹和图片名））
                       不完整的路径(仅有上一级文件夹和图片名)也可也运行 eg : 50--Celebration_Or_Party/50_Celebration_Or_Party_houseparty_50_321.jpg
            unify_img : 使用的是unify的图片还是未经过unify的origin图片(有将数据集图像重新处理并保持的想法，以此简化计算，加快训练，未实现)
            imread_type : cv.imread("img_dir", flags)中flags参数，仅用于传递参数(default:cv.IMREAD_COLOR)
            wait_time : cv.waitKey(wait_time)时间参数，仅用于传递参数(default:1000(1秒))
outputs:    None
"""
def show_key_points_in_unify_img(unify_coords, img_path, unify_img=False, imread_type=cv.IMREAD_COLOR, wait_time=1500):
    # 从img中取出矩形框选部分是通过row_col
    # 在img上画出rect或者circle是通过x_y
    # 数据集标注采用的是x_y方式标注点

    # 如果是不完整的路径(仅有上一级文件夹和图片名)则将其补全(加上文件夹所在位置)
    if not os.path.isabs(img_path):
        img_path = os.path.join(dataset_images_base_dir, img_path)

    if not unify_img:  # 传入的图片没有经过unify
        img = cv.imread(img_path, flags=imread_type)
        img_unify = img[unify_coords[row_min_rect_idx]:unify_coords[row_max_rect_idx],
                        unify_coords[col_min_rect_idx]:unify_coords[col_max_rect_idx]]
        img_unify = cv.resize(img_unify, unify_gray_image_size)
    else:  # 传入的图片经过unify，无需resize，也就不需要矩形框
        img_unify = cv.imread(img_path, flags=imread_type)
    # print(img_color_unify.shape)

    for i in range(key_points_numbers):
        cv.circle(img_unify, center=(unify_coords[2 * i], unify_coords[2 * i + 1]), radius=0, color=bgr_Pink, thickness=2)

    cv.imshow("key points", img_unify)
    cv.waitKey(wait_time)
    cv.destroyAllWindows()

# def show_key_points_in_unify_img(unify_img, unify_coords, wait_time=0):
#     unify_img_copy = unify_img.copy()
#     for i in range(key_points_numbers):
#         cv.circle(unify_img_copy, center=(unify_coords[2 * i], unify_coords[2 * i + 1]), radius=0, color=bgr_Pink, thickness=3)
#     cv.imshow("key points", unify_img_copy)
#     cv.waitKey(wait_time)
#     cv.destroyAllWindows()

"""
pass
"""
def show_train_key_points_in_unify_img(Net, Dataset, image_paths, unify_int_coords, show_train_image_idx, unify_img=False, imread_type=cv.IMREAD_COLOR, wait_time=1500):
    Net.eval()
    unify_gray_image, actual_coords = Dataset[show_train_image_idx]
    unify_gray_image = torch.reshape(unify_gray_image, (-1, 1, unify_gray_image_size[0], unify_gray_image_size[1]))  # 更改形状（添加维度）
    # unify_gray_image = unify_gray_image.to(device)  # 一定要to(device)否则报错:Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same
    train_coords = Net(unify_gray_image)
    train_coords = train_coords.detach().cpu().numpy()  # 转换为 <class 'numpy.ndarray'>  # 一定要.cpu()否则在GPU上无法正常运行
    train_coords = np.reshape(train_coords, -1)  # 拉成一维数组
    # print(train_coords.shape)  # 72
    train_coords = np.append(train_coords, unify_int_coords[show_train_image_idx][x_min_rect_idx:y_max_rect_idx+1])  # 加上矩形框大小
    train_coords = train_coords.astype(np.int32)  # 一定要数据类型转换
    train_img_path = image_paths[show_train_image_idx]
    print("eval_train_coords")
    print(train_coords)
    show_key_points_in_unify_img(unify_coords=train_coords, img_path=train_img_path, unify_img=False, imread_type=imread_type, wait_time=wait_time)
    return train_coords


"""

"""
def get_train_key_points_in_unify_img_from_camera(Net, unify_color_img, unify_gray_tensor_img):
    Net.eval()
    unify_gray_image = torch.reshape(unify_gray_tensor_img, unify_tensor_image_size)  # 更改形状（添加维度）
    unify_gray_image = unify_gray_image.to(device)
    train_coords = Net(unify_gray_image)
    train_coords = train_coords.detach().cpu().numpy()  # 转换为 <class 'numpy.ndarray'>  # 一定要.cpu()否则在GPU上无法正常运行
    train_coords = np.reshape(train_coords, -1)  # 拉成一维数组
    train_coords = train_coords.astype(np.int32)  # 一定要数据类型转换
    for i in range(key_points_numbers):
        cv.circle(unify_color_img, center=(train_coords[2 * i], train_coords[2 * i + 1]), radius=0, color=bgr_Pink, thickness=3)
    return train_coords



"""
name:       show_key_points_in_incomplete_unify_img
functional: 将不是所有关键点都在矩形框中的图片信息进行展示（仅在测试时使用，训练时并不会调用）
inputs:     unify_coords : key points coordinates after unify step
            not_unify_coords : key points coordinates before unify step（方便展示出问题）
            img_path : 图片路径
outputs:    None
"""
def show_key_points_in_incomplete_unify_img(unify_coords, not_unify_coords, img_path):
    unify_coords_shape = unify_coords.shape  # unify_coords形状
    # 标注不全的图片（人脸有些关键点在图像区域外）
    incomplete_num = 0
    temp_path = ""
    """
    for img_idx in range(unify_coords_shape[0]):
        for point_idx in range(unify_coords_shape[1]):
            if unify_coords[img_idx, point_idx] < 0:  # 有unify_coords数值为负
                incomplete_num += 1
                print("img_idx: ", img_idx, "  ", "path: ", img_path[img_idx])
                # print(unify_coords[img_idx])
                # temp_path = os.path.join(dataset_images_base_dir, img_path[img_idx])
                # show_key_points_in_unify_img(unify_coords[img_idx], temp_path)
                break
    print("共有{}张图片关键点在区域外".format(incomplete_num))  # 共有44张图片关键点在区域外
    """
    # 1035 和 2747 所有关键点在区域外
    for img_idx in (1035, 2747):
        print("img_idx: ", img_idx, "  ", "path: ", img_path[img_idx])
        print(not_unify_coords[img_idx])
        temp_path = os.path.join(dataset_images_base_dir, img_path[img_idx])
        show_key_points_and_rect_in_origin_img(not_unify_coords[img_idx], temp_path)
    """
    ( 92 , 0 ) value -4 path:  48--Parachutist_Paratrooper/48_Parachutist_Paratrooper_Parachutist_Paratrooper_48_487.jpg
    ( 107 , 0 ) value -3 path:  32--Worker_Laborer/32_Worker_Laborer_Worker_Laborer_32_23.jpg
    ( 274 , 0 ) value -4 path:  13--Interview/13_Interview_Interview_On_Location_13_179.jpg
    ( 409 , 0 ) value -1 path:  28--Sports_Fan/28_Sports_Fan_Sports_Fan_28_462.jpg
    ( 516 , 34 ) value -2 path:  36--Football/36_Football_americanfootball_ball_36_714.jpg
    ( 601 , 0 ) value -1 path:  7--Cheering/7_Cheering_Cheering_7_40.jpg
    ( 1035 , 0 ) value -66 path:  12--Group/12_Group_Group_12_Group_Group_12_839.jpg
    ( 1038 , 3 ) value -2 path:  46--Jockey/46_Jockey_Jockey_46_857.jpg
    ( 1135 , 0 ) value -1 path:  50--Celebration_Or_Party/50_Celebration_Or_Party_houseparty_50_895.jpg
    ( 1367 , 0 ) value -1 path:  4--Dancing/4_Dancing_Dancing_4_983.jpg
    ( 1451 , 1 ) value -12 path:  29--Students_Schoolkids/29_Students_Schoolkids_Students_Schoolkids_29_432.jpg
    ( 1752 , 0 ) value -1 path:  12--Group/12_Group_Group_12_Group_Group_12_122.jpg
    ( 1798 , 0 ) value -1 path:  49--Greeting/49_Greeting_peoplegreeting_49_74.jpg
    ( 1802 , 0 ) value -1 path:  28--Sports_Fan/28_Sports_Fan_Sports_Fan_28_61.jpg
    ( 2199 , 0 ) value -25 path:  2--Demonstration/2_Demonstration_Demonstration_Or_Protest_2_809.jpg
    ( 2701 , 0 ) value -8 path:  32--Worker_Laborer/32_Worker_Laborer_Worker_Laborer_32_478.jpg
    ( 2747 , 0 ) value -68 path:  28--Sports_Fan/28_Sports_Fan_Sports_Fan_28_959.jpg
    ( 2913 , 0 ) value -1 path:  44--Aerobics/44_Aerobics_Aerobics_44_189.jpg
    ( 3085 , 0 ) value -3 path:  0--Parade/0_Parade_Parade_0_592.jpg
    ( 3086 , 32 ) value -1 path:  10--People_Marching/10_People_Marching_People_Marching_2_832.jpg
    ( 3174 , 32 ) value -3 path:  2--Demonstration/2_Demonstration_Political_Rally_2_83.jpg
    ( 3187 , 32 ) value -4 path:  61--Street_Battle/61_Street_Battle_streetfight_61_282.jpg
    ( 3786 , 36 ) value -1 path:  24--Soldier_Firing/24_Soldier_Firing_Soldier_Firing_24_828.jpg
    ( 4221 , 5 ) value -1 path:  28--Sports_Fan/28_Sports_Fan_Sports_Fan_28_227.jpg
    ( 4253 , 0 ) value -10 path:  49--Greeting/49_Greeting_peoplegreeting_49_759.jpg
    ( 4266 , 0 ) value -3 path:  13--Interview/13_Interview_Interview_On_Location_13_623.jpg
    ( 4327 , 0 ) value -3 path:  18--Concerts/18_Concerts_Concerts_18_737.jpg
    ( 4574 , 32 ) value -1 path:  28--Sports_Fan/28_Sports_Fan_Sports_Fan_28_918.jpg
    ( 4852 , 0 ) value -3 path:  12--Group/12_Group_Group_12_Group_Group_12_610.jpg
    ( 4925 , 0 ) value -1 path:  4--Dancing/4_Dancing_Dancing_4_22.jpg
    ( 4994 , 0 ) value -8 path:  2--Demonstration/2_Demonstration_Political_Rally_2_803.jpg
    ( 5551 , 0 ) value -2 path:  13--Interview/13_Interview_Interview_On_Location_13_765.jpg
    ( 5681 , 32 ) value -1 path:  35--Basketball/35_Basketball_playingbasketball_35_476.jpg
    ( 5701 , 32 ) value -7 path:  29--Students_Schoolkids/29_Students_Schoolkids_Students_Schoolkids_29_531.jpg
    ( 5721 , 0 ) value -1 path:  50--Celebration_Or_Party/50_Celebration_Or_Party_houseparty_50_33.jpg
    ( 5926 , 0 ) value -3 path:  2--Demonstration/2_Demonstration_Demonstrators_2_398.jpg
    ( 6007 , 0 ) value -2 path:  12--Group/12_Group_Group_12_Group_Group_12_35.jpg
    ( 6076 , 0 ) value -2 path:  41--Swimming/41_Swimming_Swimmer_41_1039.jpg
    ( 6087 , 0 ) value -7 path:  12--Group/12_Group_Large_Group_12_Group_Large_Group_12_219.jpg
    ( 6360 , 5 ) value -2 path:  47--Matador_Bullfighter/47_Matador_Bullfighter_matadorbullfighting_47_561.jpg
    ( 6554 , 0 ) value -3 path:  7--Cheering/7_Cheering_Cheering_7_805.jpg
    ( 6637 , 0 ) value -12 path:  58--Hockey/58_Hockey_icehockey_puck_58_580.jpg
    ( 6805 , 0 ) value -1 path:  14--Traffic/14_Traffic_Traffic_14_620.jpg
    ( 7120 , 38 ) value -1 path:  50--Celebration_Or_Party/50_Celebration_Or_Party_houseparty_50_853.jpg
    """


# WFLW数据集路径检查
def check_path():
    # print(separate_bar, "WFLW数据集路径检查开始", separate_bar)
    if os.path.exists(dataset_images_base_dir):
        print("WFLW数据集图像路径： ", dataset_images_base_dir)
        print("成功找到WFLW数据集图像路径")
    else:
        print("WFLW数据集图像路径有误，请对Consts.py文件进行修改")
        return False

    if os.path.isfile(train_dataset_annotation_dir):
        print("WFLW数据集标注路径： ", train_dataset_annotation_dir)
        print("成功找到WFLW数据集标注路径")
    else:
        print("WFLW数据集标注路径有误，请对Consts.py文件进行修改")
        return False
    # print(separate_bar, "WFLW数据集路径检查完毕", separate_bar)
    return True


# WFLW数据集信息显示
def inform_dataset_basic(_origin_anno_size, _simple_anno_size, DatasetType="train"):
    # print(separate_bar, "数据集相关信息如下", separate_bar)
    if DatasetType == "train":
        TypePrefix = "训练"
        dataset_name = train_dataset_name
    elif DatasetType == "test":
        TypePrefix = "测试"
        dataset_name = test_dataset_name
    else:
        print("Dataset Type can not be matched.")
        exit(-1)

    print("数据集名称： ", dataset_name)
    print("原始" + TypePrefix + "数据标注shape：", _origin_anno_size)
    print("简化" + TypePrefix + "数据标注shape：", _simple_anno_size)
    # print(separate_bar, "输出数据集信息结束", separate_bar)


def testGPU():
    # print(separate_bar, "开始测试GPU", separate_bar)
    if torch.cuda.is_available():
        print("GPU is available. Use GPU.")
        # print(separate_bar, "GPU测试结束", separate_bar)
        return True
    else:
        print("Only CPU is available. Use CPU.")
        # print(separate_bar, "GPU测试结束", separate_bar)
        return False


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
    simple_annotation_p1 = origin_annotation[:, left_eye_start_ * 2: inner_lip_end_ * 2 + 2]  # 眼睛嘴唇的关键点正好连续
    simple_annotation_p2 = origin_annotation[:, x_min_rect_idx_: y_max_rect_idx_ + 1]  # 脸部矩形的两个点
    simple_annotation_p3 = origin_annotation[:, img_relative_root_idx_]  # 对应图像路径
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
    pass
