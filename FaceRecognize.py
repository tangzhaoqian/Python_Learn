
def facerecognize():
    import cv2
    import mediapipe as mp
    import time
    cap = cv2.VideoCapture(0)
    faceR = mp.solutions.face_detection
    facedetection = faceR.FaceDetection()
    mpdraw = mp.solutions.drawing_utils

    pTime = 0
    cTime = 0

    while True:

        success, img = cap.read()

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = facedetection.process(imgRGB)
        if results.detections:
            for id, facelms in enumerate(results.detections):
                #mpdraw.draw_detection(img, facelms)
                bboxc = facelms.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxc.xmin * iw), int(bboxc.ymin * ih), int(bboxc.width * iw), int(bboxc.height * ih)
                cv2.rectangle(img, bbox, (255, 0, 255), 2)
                cv2.putText(img, f' {int(facelms.score[0]*100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
        cv2.imshow("face detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
