# 1950083 自动化 刘智宇
import torch
from torchvision import transforms as tf
import cv2 as cv
import numpy as np

import ToolFunction
from Consts import *
from KeyPointNet import KeyPointNet

def face_detection(origin_image, show_result=False):
    origin_gary_image = cv.cvtColor(src=origin_image, code=cv.COLOR_BGR2GRAY)
    face_detect = cv.CascadeClassifier(r"OpenCVModel/haarcascade_frontalface_alt2.xml")
    faces_pos = face_detect.detectMultiScale(image=origin_gary_image)

    faces_num = len(faces_pos)

    if faces_num == 0:
        return 0, 0, 0, 0, faces_num
    else:
        max_face_w = 0; max_face_h = 0; max_face_x = 0; max_face_y = 0
        max_face_area = 0
        for x, y, w, h in faces_pos:  # 只找出最大的脸
            if w * h > max_face_area:
                max_face_area = w * h
                max_face_w = w; max_face_h = h; max_face_x = x; max_face_y = y
        # cv.rectangle(img=origin_image, pt1=(max_face_x, max_face_y),
        #              pt2=(max_face_x + max_face_w, max_face_y + max_face_h), thickness=2, color=(0, 255, 0))
        # print(type(gray_unify_face))  # <class 'numpy.ndarray'>

    if show_result:
        cv.imshow("result", origin_image)
    return max_face_x, max_face_y, max_face_w, max_face_h, faces_num


if __name__ == "__main__":
    # 导入模型
    print("\n" + "导入训练好的模型：")
    print(use_model_name)
    net = KeyPointNet(NetChoice=net_choice)
    net.load_state_dict(torch.load(use_model_name))
    net.to(device)  # 模型转入device
    print("模型导入成功" + "\n")

    totensor_transform = tf.ToTensor()

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
            face_x, face_y, face_w, face_h, face_num = face_detection(origin_image=frame, show_result=False)
            if face_num == 0:
                print("未检测到驾驶员面部!")
                frame_cut = np.ones(unify_color_image_size)  # print(frame_cut.shape)  # (224, 224, 3)
                frame_cut_gray = np.ones(unify_gray_image_size)
            else:
                print("检测到驾驶员面部")
                frame_cut = frame[face_y: face_y + face_h, face_x: face_x + face_w]  # row-col和x-y的区别
                frame_cut = cv.resize(src=frame_cut, dsize=unify_image_size)  # 调整图片大小（只改变行列，通道数不改变）
                frame_cut_gray = cv.cvtColor(src=frame_cut, code=cv.COLOR_BGR2GRAY)
                # 不需要在这里转tensor，后续函数会自己转
                frame_cut_gray_tensor = totensor_transform(frame_cut_gray)  # print(frame_cut_gray_tensor.shape)  torch.Size([1, 224, 224])
                ToolFunction.get_train_key_points_in_unify_img_from_camera(Net=net, unify_color_img=frame_cut, unify_gray_tensor_img=frame_cut_gray_tensor)

            cv.imshow("Driver Face", frame_cut)
            waitkey = cv.waitKey(5)
            if waitkey == ord('q') or waitkey == 27:  # 退出
                break
        Camera.release()
        cv.destroyAllWindows()
