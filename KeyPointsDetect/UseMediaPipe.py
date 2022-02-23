# 1950083 自动化 刘智宇
import cv2
import mediapipe as mp
import cv2 as cv
import time
from Consts import *


if __name__ == "__main__":
    pTime = 0
    cTime = 0

    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
    drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Can't open the camera or video.")
        exit(-1)
    while True:
        ret, frame = cap.read()

        frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = faceMesh.process(image=frameRGB)

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_FACE_OVAL, drawSpec, drawSpec)

                for idx, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = frame.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    print(id, x, y)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv.putText(frame, f"FPS:{int(fps)}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, bgr_Red, 3)
        cv.imshow("Frame", frame)
        if cv.waitKey(2) == ord("q"):
            break
    cv.destroyAllWindows()

