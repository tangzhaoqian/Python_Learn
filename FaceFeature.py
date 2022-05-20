def facefeature():
    import cv2
    import mediapipe as mp
    import time

    cTime = 0
    pTime = 0

    mpdraw = mp.solutions.drawing_utils
    mpfacemesh = mp.solutions.face_mesh
    facefd = mpfacemesh.FaceMesh(max_num_faces=2)

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = facefd.process(imgRGB)
        if results.multi_face_landmarks:
            for facelms in results.multi_face_landmarks:
                mpdraw.draw_landmarks(img, facelms, mpfacemesh.FACEMESH_CONTOURS)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f'FPS:{int(fps)}',(20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0))
        cv2.imshow("face feature detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
