# 1950083 自动化 刘智宇
import os
import ToolFunction
from WFLWdataset import WFLW_Dataset
from Consts import *
from KeyPointNet import KeyPointNet
if __name__ == "__main__":
    # 模型列表
    """
    Models/GPU_WithResNet18_SGDOptim_MAELoss_Epoch10_BatchSize50_LR0.01_MaxBatchLoss11.956532.pth
    Models/GPU_WithResNet18_SGDOptim_MAELoss_Epoch6_BatchSize50_LR0.01_MaxBatchLoss16.683079.pth
    Models/GPU_WithResNet34_SGDOptim_MAELoss_Epoch6_BatchSize5_LR0.01_MaxBatchLoss23.802553.pth
    """
    model_complete_name = r"Models/GPU_WithResNet34_SGDOptim_MAELoss_Epoch6_BatchSize5_LR0.01_MaxBatchLoss23.802553.pth"
    # ---------------------- 网络模型验证 ----------------------
    print(separate_bar*2, "网络模型验证：", separate_bar*2)
    print(model_complete_name)

    simple_annotation, origin_annotation_size, simple_annotation_size = read_annotation()
    simple_coords_annotation_float = simple_annotation[:, 0:img_relative_root_idx]  # 只取出坐标标注
    simple_coords_annotation_int = simple_coords_annotation_float.astype(np.int32)  # 讲坐标标注转为整形方便opencv处理
    img_paths_annotation = simple_annotation[:, -1]  # 只取出文件路径标注
    unify_coords = ToolFunction.unify_img_coords_annotation(simple_coords_annotation_int, img_paths_annotation)
    train_dataset = WFLW_Dataset(unify_coords_anno=unify_coords, img_paths_anno=img_paths_annotation)

    eval_net = KeyPointNet(net_choice)
    eval_net.load_state_dict(torch.load(model_complete_name))

    # eval_net = torch.load(model_complete_name)
    # print(next(eval_net.parameters()).is_cuda)  # True
    # for _, param in enumerate(eval_net.named_parameters()):
    #     print(param[0])
    #     print(param[1])
    #     print('----------------')
    # exit()

    print("\n" + "网络打开成功：")
    # eval_net.eval()
    #
    # print(eval_net)

    # while True:
    eval_img_idx = input("Index of the Image and the Key Points you want to check (0-7499) : ")
    # print(type(eval_img_idx))  # <class 'str'>
    eval_img_idx = int(eval_img_idx)
    if eval_img_idx < 0 or eval_img_idx > 7499:
        eval_img_idx = 1234

    print("eval_actual_coords")
    print(unify_coords[eval_img_idx])

    eval_train_coords = ToolFunction.show_train_key_points_in_unify_img(Net=eval_net, Dataset=train_dataset, image_paths=img_paths_annotation,
                                                                        unify_int_coords=simple_coords_annotation_int,
                                                                        show_train_image_idx=eval_img_idx, wait_time=0)



