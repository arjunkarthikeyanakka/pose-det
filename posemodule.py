import time
import mediapipe as mp
import cv2
print("setup complete good to go!")
x=mp.solutions.pose

class whatPose():

    def __init__(self,mode=False,comp=1,smooth=True,min_det=0.5,min_track=0.5):
        self.mode=mode
        self.complexity=comp
        self.smooth_landmarks=smooth
        self.enable_segmentation=False
        self.smooth_segmentation=smooth
        self.min_detection_confidence=min_det
        self.min_tracking_confidence=min_track
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode,self.complexity,self.smooth_landmarks,self.enable_segmentation,self.smooth_segmentation,self.min_detection_confidence,self.min_tracking_confidence)

    def findPose(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        return img

    def getPoints(self,img,draw=True):
        lmlist = []
        if self.results.pose_landmarks:
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape
                cx ,cy = int(lm.x*w) , int(lm.y*h)
                lmlist.append([id,cx,cy])
                if draw and id==14: #this will track the left elbow of the person in the video footage...
                    cv2.circle(img,(cx,cy),5,(0,0,0),cv2.FILLED)
        return lmlist




def main():
    cap = cv2.VideoCapture(0)
    ptime = 0
    Sample = whatPose()
    while 1:
        success, img = cap.read()
        img = Sample.findPose(img)
        landmark_list = Sample.getPoints(img)   #this list will have the co-ordinates of all the 33 points of body after every frame...
        #we can print coordinates of some specific point and track its movement using color command
        #cv2.circle(img,(landmark_list[0][1],landmark_list[0][2]),15,(255,255,0),cv2.FILLED)
        ctime = time.time()
        fps=0
        if ctime-ptime:
            fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(img, str(int(fps)), (120, 50), 3, cv2.FONT_HERSHEY_DUPLEX, (0, 255, 0), 3)
        cv2.imshow("With great power comes great responsibility",img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
