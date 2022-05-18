import cv2
import mediapipe as mp
import time


def poseestimation():
    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    pTime = 0
    cTime = 0

    cap = cv2.VideoCapture("video/4506.mp4")  # detect pepople's pos from the video
#    cap = cv2.VideoCapture(0) # detect the pose of user 

    while True:

        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(round(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("pose_estimation", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
