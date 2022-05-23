import cv2
import mediapipe as mp
import time
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

def VolControl():
    cap = cv2.VideoCapture(0)
    cTime = 0
    pTime = 0
    vol = 0
    # the hand tracking
    handtr = mp.solutions.hands
    hands = handtr.Hands()
    mpdraw = mp.solutions.drawing_utils

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volrange = volume.GetVolumeRange()
    minvol = volrange[0]
    maxvol = volrange[1]

    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        lmList = []
        if results.multi_hand_landmarks:
            hand_landmark = results.multi_hand_landmarks[0]
            for id, lms in enumerate(hand_landmark.landmark):
                h, w, c = img.shape
                cx, cy = int(lms.x * w), int(lms.y * h)
                lmList.append([id, cx, cy])

            for handlms in results.multi_hand_landmarks:
                mpdraw.draw_landmarks(img, handlms, handtr.HAND_CONNECTIONS)

        if len(lmList) != 0:
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            length = math.hypot(x2-x1, y2-y1)

            vol = np.interp(length, [50, 300], [minvol, maxvol])
            volume.SetMasterVolumeLevel(vol, None)
            
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f'fps:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
        cv2.imshow("volume-hand-tracking", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

