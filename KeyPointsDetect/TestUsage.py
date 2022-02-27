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
    img = cv.imread(r"Models/GPU_WithResNet18_NotPretrain_SGDOptim_MAELoss_Epoch5_BatchSize5_LR8e-05_LastAverageLoss134.12452697753906.pth_GraphicDisplay.jpg",
                    flags=cv.IMREAD_COLOR)
    print(img.shape)
