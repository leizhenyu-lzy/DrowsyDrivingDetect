# 1950083 自动化 刘智宇
import os
import cv2 as cv
import numpy as np

"""
name:       
functional: 
inputs:     
outputs:  
"""
def train_recognizer(faces_dir, detector, whether_train=True):
    print("开始训练 Driver Recognizer")
    idxs, faces, labels = get_image_and_label(faces_dir=faces_dir, detector=detector)
    if whether_train:
        recognizer = cv.face.LBPHFaceRecognizer_create()  # LBPH不受光照影响（EigenFaces、FisherFaces、LBPH）
        recognizer.train(faces, np.array(idxs))
        recognizer.write(r"driver_recognizer.yml")
    print("Driver Recognizer保存完成")
    return idxs, faces, labels

def get_image_and_label(faces_dir, detector):
    idxs = []
    faces = []  # 驾驶员人脸列表
    labels = []  # 驾驶员信息列表
    faces_paths_list = []  # 驾驶员人脸图片路径

    print("读取到以下图片")
    for idx, imgname in enumerate(os.listdir(faces_dir)):
        idxs.append(idx)
        labels.append(imgname)
        faces_paths_list.append(os.path.join(faces_dir, imgname))
        print(faces_paths_list[idx])

    for idx, face_path in enumerate(faces_paths_list):
        image = cv.imread(filename=face_path, flags=cv.IMREAD_GRAYSCALE)
        faces_pos = detector.detectMultiScale(image)
        face_pos = faces_pos[0]
        x = face_pos[0]; y = face_pos[1]; w = face_pos[2]; h = face_pos[3]

        face_cut = image[y: y + h, x: x + w]
        face_unify = cv.resize(src=face_cut, dsize=[224, 224])
        faces.append(face_unify)
        # print(idx, face_unify.shape)

    return idxs, faces, labels


"""
name:       
functional: 
inputs:     
outputs:  
"""
def use_recognizer(recognizer, detector, labels):
    camera = cv.VideoCapture(0)
    if not camera.isOpened():
        print("Can't open the camera.")
        exit(-1)
    print("Camera is open.")
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Can't receive the frame.")
            break
        frame_gray = cv.cvtColor(src=frame, code=cv.COLOR_BGR2GRAY)
        faces_pos = detector.detectMultiScale(image=frame_gray)
        face_pos = faces_pos[0]
        x = face_pos[0]; y = face_pos[1]; w = face_pos[2]; h = face_pos[3]
        face_cut_gray = frame_gray[y: y + h, x: x + w]
        face_cut_gray_unify = cv.resize(src=face_cut_gray, dsize=(224, 224))
        cv.imshow("test_camera", face_cut_gray_unify)

        idx, confidence = recognizer.predict(face_cut_gray_unify)
        print("driver:", labels[idx], "  ", "confidence: ", confidence)
        waitkey = cv.waitKey(5)
        if waitkey == ord('q') or waitkey == 27:  # 退出
            break

    cv.destroyAllWindows()
    camera.release()



"""
name:       
functional: 
inputs:     
outputs:  
"""
def take_picture():
    pass


if __name__ == "__main__":
    """
        OpenCV Models
        1. r"D:\Project\DrowsyDrivingDetect\KeyPointsDetect\OpenCVModel\haarcascade_frontalface_alt.xml"
        2. r"D:\Project\DrowsyDrivingDetect\KeyPointsDetect\OpenCVModel\haarcascade_frontalface_alt2.xml"
        3. r"D:\Project\DrowsyDrivingDetect\KeyPointsDetect\OpenCVModel\haarcascade_frontalface_default.xml"
        """
    # 一些常量
    face_detector = cv.CascadeClassifier(r"D:\Project\DrowsyDrivingDetect\KeyPointsDetect\OpenCVModel\haarcascade_frontalface_alt2.xml")
    driverfaces_dir = r"DriverFaces"
    driver_recognizer = cv.face.LBPHFaceRecognizer_create()
    driver_recognizer.read('driver_recognizer.yml')

    # 训练分类器
    idx_list, faces_list, labels_list = train_recognizer(faces_dir=driverfaces_dir, detector=face_detector, whether_train=True)

    # 试用recognizer
    use_recognizer(recognizer=driver_recognizer, detector=face_detector, labels=labels_list)
    # 打开摄像头进行验证


