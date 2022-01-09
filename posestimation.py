import time
import cv2
import mediapipe as mp
print("setup complete , good to go!")

cap = cv2.VideoCapture("pose3levi.mp4")

'''
pose function has params : 

            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5):
            
            ***IMP : If the detection confidence crosses 0.5 then the image can be detected so we check for 
                     tracking confidence. Whenever the tracking confidence gets below 0.5 , we go back to 
                     detecting the image.
                     
            ** : if static image mode is put True ----  then , it will always detect from model. 
                 else : the above IMP point happens , detecting and tracking will happen , based on the values.
                        This way we do not use the heavy model , so the working of our model will be simple and fast.


'''


mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

prevTime = 0

while 1:

    success, img = cap.read()

    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)  #this will convert the image from bgr to rgb

    results = pose.process(imgRGB)  #this will process the RGBimage

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id,lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            cx , cy = int(lm.x*w) , int(lm.y*h)
            cv2.circle(img,(cx,cy),5,(0,255,0),cv2.FILLED)

    currTime = time.time()

    fps=0

    fps = 1/(currTime-prevTime)

    prevTime = currTime

    # this will print frames per second on the coordinates (10,70) on the window
    cv2.putText(img, str(int(fps)), (100, 70), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 0, 255), 3)

    cv2.imshow("Pose estimator", img)

    cv2.waitKey(1)