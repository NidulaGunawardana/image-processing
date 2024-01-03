import cv2
import mediapipe as mp
import time

# codes are here https://www.computervision.zone/courses/advance-computer-vision-with-python/

def rescaleFrame(frame,scale=0.75):
    """
    This method will work for any kind of video photo or live video
    """
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    
    dimentions = (width,height)
    
    return cv2.resize(frame,dimentions,interpolation=cv2.INTER_AREA)

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
cap = cv2.VideoCapture('C:/Users/nidul/Desktop/ACCA Folder My/image processing/Deep image clasifire/Body Tracking/Video/2.mp4')
pTime = 0
while True:
    success, frame = cap.read()
    img = rescaleFrame(frame,0.5)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    
    cv2.imshow("Image", img)
    
    if cv2.waitKey(10) & 0xFF==ord('d'):
        break