import time
import mediapipe as mp
import cv2
import posemodule as pm

cap = cv2.VideoCapture(0)
Sample = pm.whatPose()
ptime=0
while 1:
    success , img = cap.read()
    img = Sample.findPose(img)
    landmark_list = Sample.getPoints(img)
    ctime = time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    cv2.putText(img,str(int(fps)),(120,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,3,(0,255,0),3)
    cv2.imshow("Pose-Finder",img)
    cv2.waitKey(1)