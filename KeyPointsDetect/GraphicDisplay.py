# 1950083 自动化 刘智宇
import numpy as np
import matplotlib.pyplot as plt


"""
name:       
functional: 
inputs:     net_name    : 网络名称，用于图片保存
            train_loss  : 列表
            test_loss   : 列表
            step        : 列表
            train_loss、test_loss、step三个数据应该保证是同维度的
outputs:    
"""
def GraphicDisplayLoss(net_name, train_loss_list, test_loss_list, batch_step_list, epoch_step_list):
    # 先进行数据检查
    train_loss_len = len(train_loss_list)
    test_loss_len = len(test_loss_list)
    batch_step_list_len = len(batch_step_list)
    epoch_step_list_len = len(epoch_step_list)
    if not train_loss_len == batch_step_list_len:
        print("train_loss_len{} 和 batch_step_list_len{} 长度不匹配".format(train_loss_len, batch_step_list_len))
        exit(-1)
    if not test_loss_len == epoch_step_list_len:
        print("test_loss_len{} 和 epoch_step_list_len{} 长度不匹配".format(test_loss_len, epoch_step_list_len))
        exit(-1)

    # 通过检查后，进行画图
    plt.figure(num="GraphicShow")
    plt.title(net_name)
    train_loss_fig = plt.plot(batch_step_list, train_loss_list, color="b", linestyle="-", marker='+', label="train loss")
    test_loss_fig = plt.plot(epoch_step_list, test_loss_list, color="r", linestyle="-.", marker='+', label="test loss")

    plt.xlabel("Train Steps")
    plt.ylabel("Batch Loss")
    plt.legend()  # 展示图例

    # 图片保存及展示
    fig_name = net_name + "_GraphicDisplay.jpg"
    plt.savefig(fig_name)
    plt.show()


if __name__ == "__main__":
    a = [1, 2, 3, 4, 5]
    b = [5, 4, 3]
    bt = [11, 22, 33, 44, 55]
    tt = [66, 77, 88]

    GraphicDisplayLoss("abcde", a, b, bt, tt)

    pass
