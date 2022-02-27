# 1950083 自动化 刘智宇
import torch
from torch import nn, optim
import torchvision.models
from Consts import *

with_alexnet = torchvision.models.alexnet(pretrained=net_pretrain)
with_resnet18 = torchvision.models.resnet18(pretrained=net_pretrain)
with_resnet34 = torchvision.models.resnet34(pretrained=net_pretrain)
with_resnet50 = torchvision.models.resnet50(pretrained=net_pretrain)
with_googlenet = torchvision.models.googlenet(pretrained=net_pretrain)
with_vgg16 = torchvision.models.vgg16(pretrained=net_pretrain)


def GetLoss(loss_idx=lossfunc_choice):
    if loss_idx == mse_idx:
        return nn.MSELoss()
    elif loss_idx == mae_idx:
        return nn.L1Loss()
    elif loss_idx == cross_entropy_idx:
        return nn.CrossEntropyLoss()
    elif loss_idx == huber_idx:
        return nn.HuberLoss()
    else:
        print("The loss function choice is not valid.")
        exit(1)

def GetOptimizer(optim_idx, parameter, LR=train_lr):
    if optim_idx == sgd_idx:
        return optim.SGD(parameter, LR)
    elif optim_idx == adam_idx:
        return optim.Adam(parameter, LR)
    else:
        print("The optimizer choice is not valid.")
        exit(1)

class KeyPointNet(nn.Module):
    def __init__(self, NetChoice):
        super().__init__()
        self.NetChoice = NetChoice
        
        if NetChoice == kpnet_idx:
            self.net = nn.Sequential(
                nn.MaxPool2d(kernel_size=2),  # 1*112*112
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=11, padding=5),  # 1*112*112
                nn.MaxPool2d(kernel_size=2),  # 1*56*56
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(num_features=1),  # 1*56*56
                nn.Conv2d(in_channels=1, out_channels=3, kernel_size=7, padding=3),  # 3*56*56
                nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, padding=1),  # 5*56*56
                nn.BatchNorm2d(num_features=5),  # 5*56*56
                nn.MaxPool2d(kernel_size=4),  # 5*14*14
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=5, out_channels=3, kernel_size=3, padding=1),  # 3*14*14
                nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=0),  # 1*12*12
                nn.BatchNorm2d(num_features=1),  # 1*12*12
                nn.Flatten(),  # 144
                nn.ReLU(inplace=True),
                nn.Linear(in_features=144, out_features=72, bias=True),  # 72
            )
            if not net_train_bias:  # 不对线性层的bias进行训练
                for _, param in enumerate(self.net.named_parameters()):
                    if param[0] == "15.bias":
                        param[1].requires_grad = False

        elif NetChoice == with_alexnet_idx:  # 输入输出参数维数已匹配
            self.net = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1),
                with_alexnet,  # output_shape: 1000
                nn.Linear(in_features=1000, out_features=key_points_numbers*2, bias=net_train_bias)
            )
        elif NetChoice == with_resnet18_idx:
            self.net = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1),
                with_resnet18,
                nn.BatchNorm1d(num_features=1000),
                nn.Linear(in_features=1000, out_features=key_points_numbers*2, bias=net_train_bias),
            )
            if not net_train_bias:  # 不对线性层的bias进行训练
                for _, param in enumerate(self.net.named_parameters()):
                    if param[0] == "2.bias":
                        param[1].requires_grad = False  # 不对bias进行训练
        elif NetChoice == with_resnet34_idx:
            self.net = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1),
                with_resnet34,
                nn.BatchNorm1d(num_features=1000),
                nn.Linear(in_features=1000, out_features=key_points_numbers * 2, bias=net_train_bias),
            )
            if not net_train_bias:  # 不对线性层的bias进行训练
                for _, param in enumerate(self.net.named_parameters()):
                    if param[0] == "3.bias":
                        param[1].requires_grad = False  # 不对bias进行训练
        elif NetChoice == with_resnet50_idx:
            self.net = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1),
                with_resnet50,
                nn.Linear(in_features=1000, out_features=key_points_numbers * 2, bias=net_train_bias),
            )
            # for _, param in enumerate(self.net.named_parameters()):
            #     # print(param[0])
            #     # print(param[1])
            #     if param[0] == "2.bias":
            #         param[1].requires_grad = False  # 不对bias进行训练
            #         # print(param[0])
            #         # print(param[1])
        elif NetChoice == with_googlenet_idx:
            self.net = nn.Sequential(

            )
        elif NetChoice == with_vgg16_idx:
            self.net = nn.Sequential(

            )

    def forward(self, Input):
        Output = self.net(Input)
        # print(Input)
        # print(Output)
        # input:    torch.Size([1, 1, 224, 224])    torch.float32
        # output:   torch.Size([1, 72])             torch.float32
        return Output


if __name__ == "__main__":
    # 检查搭建的网络正确性
    kpnet_test = KeyPointNet(NetChoice=net_choice)
    test_input = torch.ones((1, 1, 224, 224))
    test_output = kpnet_test(test_input)
    print("output_shape", test_output.shape)


