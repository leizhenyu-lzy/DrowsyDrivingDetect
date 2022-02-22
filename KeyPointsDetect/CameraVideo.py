# 1950083 自动化 刘智宇
import torch
import cv2 as cv
import numpy as np
from Consts import *
from KeyPointNet import KeyPointNet

if __name__ == "__main__":
    # 导入模型
    print("\n" + "导入训练好的模型")
    print(use_model_name)
    net = KeyPointNet(NetChoice=net_choice)
    net.load_state_dict(torch.load(use_model_name))
    print("模型导入成功")

    net.eval()
    with torch.no_grad():
        # 相机演示
        Camera = cv.VideoCapture(0)
        if not Camera.isOpened():
            print("Can't open the camera.")
            exit(-1)
        while True:
            ret, frame = Camera.read()
            if not ret:
                print("Can't receive the frame.")
                break
            # 进行输出
            frame_device = frame.to(device)
            frame_coords = net(frame_device)

            frame_coords_int = frame_coords.int()

            # 不行，先要找到人脸

            cv.imshow('CameraFrame', frame)
            if cv.waitKey(5) == ord('q'):  # 退出
                break
        Camera.release()
        cv.destroyAllWindows()
