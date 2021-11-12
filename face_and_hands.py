# Import Libraries
import cv2
import time
import mediapipe as mp
import numpy as np


# 손인식 개수, 학습된 제스쳐
max_hands = 1
gesture = {
    0: 'fist', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
    6: 'six', 7: 'rock', 8: 'spiderman', 9: 'yeah', 10: 'ok',
}

# 사용할 제스쳐 rock paper scissors = rps
rps_gesture = {0: 'rock', 5: 'paper', 9: 'scissors'}

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=max_hands,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Grabbing the Holistic Model from Mediapipe
# Initializing the Model
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)

# Gesture recognition model
# file = (99, 21)
# angle = (99, 20) column_indexes = 0~20
# label = (99, 1)  column_index   = 21
# opencv 에서 제공되는 machine learning 모델 중에서 knn 을(KNearest_create())사용
# knn.train 을 이용해서 학습
file = np.genfromtxt('data/gesture_train.csv', delimiter=',')
angle = file[:, :-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

# Initializing the drawing utils for drawing the facial landmarks on image
mp_drawing = mp.solutions.drawing_utils

# (0) in VideoCapture is used to connect to your computer's default camera
capture = cv2.VideoCapture(0)

# Initializing current time and precious time for calculating the FPS
previousTime = 0
currentTime = 0

# load overlay images
overlay = cv2.imread('samples/ryan_transparent.png', cv2.IMREAD_UNCHANGED)
overlay1 = cv2.imread('samples/btss.png', cv2.IMREAD_UNCHANGED)
overlay2 = cv2.imread('samples/circle.png', cv2.IMREAD_UNCHANGED)


# overlay function
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    try:
        bg_img = background_img.copy()
        # convert 3 channels to 4 channels
        if bg_img.shape[2] == 3:
          bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

        if overlay_size is not None:
          img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

        b, g, r, a = cv2.split(img_to_overlay_t)

        mask = cv2.medianBlur(a, 5)

        h, w, _ = img_to_overlay_t.shape
        roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

        img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
        img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

        bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)

        # convert 4 channels to 4 channels
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

        return bg_img
    except Exception: return background_img


while capture.isOpened():
    # capture frame by frame
    ret, frame = capture.read()

    # resizing the frame for better view
    frame = cv2.resize(frame, (800, 600))

    # 좌우 반전 추가
    # resizing = frame -> flip, bgr2rgb = image 변수에 저장
    image = cv2.flip(frame, 1)
    # Converting the from from BGR to RGB
    # opencv: BGR
    # Mediapipe: RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 손이미지 처리하기위해 변환된 image 를 hands.process 통해 result 변수에 저장
    result = hands.process(image)  # hands

    # Making predictions using holistic model
    # To improve performance, optionally mark the image as not writable to
    # pass by reference.
    image.flags.writeable = False
    results = holistic_model.process(image)  # holistic 을 이용해서 얼굴을 인식
    image.flags.writeable = True

    #########################################################################
    # writeable 사용하는 이유? 방법
    # Make an array immutable(read - only)
    # Z = np.zeros(10)
    # Z.flags.writeable = False
    # Z[0] = 1
    #
    # ValueError: assignment
    # destination is read - only
    #
    # Z.flags.writeable = False 로 설정함으로써 변수를 변경하는 것을 막았다.
    #########################################################################

    # Converting back the RGB image to BGR
    # 원래 캠의 색을 표현하기 위해 미디어파이프에서 사용한 RGB 를 다시 BGR 로 변환
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Drawing the Facial Landmarks
    # FACE_CONNECTIONS -> FACEMESH_TESSELATION 변경됨
    # 윤곽선 : FACEMESH_CONTOURS
    # https://github.com/google/mediapipe
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_TESSELATION,
        mp_drawing.DrawingSpec(
            color=(255, 0, 255),
            thickness=1,
            circle_radius=1
        ),
        mp_drawing.DrawingSpec(
            color=(0, 255, 255),
            thickness=1,
            circle_radius=1
        )
    )
    # hands 인식은 mediapipe hands 를 사용함 holistic 에서도 같은 인식이 가능하나 학습한 것을 적용하는 법을 아직 모르겠음
    # 그래서 두모델을 사용해서 따로 인식함
    # # Drawing Right hand Land Marks
    # mp_drawing.draw_landmarks(
    #     image,
    #     results.right_hand_landmarks,
    #     mp_holistic.HAND_CONNECTIONS
    # )
    #
    # # Drawing Left hand Land Marks
    # mp_drawing.draw_landmarks(
    #     image,
    #     results.left_hand_landmarks,
    #     mp_holistic.HAND_CONNECTIONS

    # 랜드마크확인해서 필터입혀 보려고 한건데 안됨
    # if results.face_landmarks is not None:
    #     for fl in results.face_landmarks:
    #         f_joint = np.zeros((486, 3))
    #         for i, fll in enumerate(fl.landmark):
    #             print(f_joint = [fll.x, fll.y, fll.z])

    if result.multi_hand_landmarks is not None:
        rps_result = []
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]
                # joint = [ 0번 landmark[x ,y ,z],
                #           1번 landmark[x ,y ,z],
                #           2번 landmark[x ,y ,z],
                #
                #          21번 landmark[x ,y ,z]]
                # joint 에 인덱스를 넣어준다

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:]  # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:]  # Child joint
            v = v2 - v1  # [20,3]
            # Normalize v
            # v 는 새로운 행렬벡터임 joint 가 아님
            # v 는 열이 하나인 series 객체임? 행과열을 갖는 행렬로 변환하기 위해 np.newaxis를 사용해서 (21, 1)로 변환
            # norm = (x^2 + y^2 + z^2)^(1/2)
            # axis=1 -> 같은 행에서만 계산해야함
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            # A dot B = |A||B|cos theta -> A dot B = cos theta -> (A dot B)/cos = theta
            # 1/cos -> arcos
            # norm 해줬기 때문에 a, b벡터의 크기는 1이다
            # 위치에 관계없이 포즈를 인식시키기 위해서 각도를 이용함
            angle = np.arccos(np.einsum('nt,nt->n',
                                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

            angle = np.degrees(angle)  # Convert radian to degree

            # Inference gesture
            # 학습된 모델을 사용하여 제스쳐 추측
            data = np.array([angle], dtype=np.float32)
            ret, knn_results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(knn_results[0][0])

# # 손인식 개수, 학습된 제스쳐
# max_hands = 1
# gesture = {
#     0: 'fist', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
#     6: 'six', 7: 'rock', 8: 'spiderman', 9: 'yeah', 10: 'ok',
# }
#
# # 사용할 제스쳐 rock paper scissors = rps
# rps_gesture = {0: 'rock', 5: 'paper', 9: 'scissors'}

            # Draw gesture result
            if idx in rps_gesture.keys():
                # (y, x) ????
                # org: text 의 좌표
                org = (int(res.landmark[0].x * image.shape[1]), int(res.landmark[0].y * image.shape[0]))
                cv2.putText(image, text=rps_gesture[idx].upper(), org=(org[0], org[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                rps_result.append({
                    'rps': rps_gesture[idx],
                    'org': org
                })

                #rps_result = [{'rps': 'rock', 'org': (x, y)}]

            # mp_drawing.draw_landmarks(image, res, mp_hands.HAND_CONNECTIONS)

            mp_drawing.draw_landmarks(image, res, mp_hands.HAND_CONNECTIONS)

            # depends on pose shows different image
            if len(rps_result) >= 1:
                pic = None
                text = ''

                if rps_result[0]['rps'] == 'rock':
                    text = 'face : ryan'
                    pic = 1
                # print(winner)
                # print(text)

                elif rps_result[0]['rps'] == 'paper':
                    text = 'face : bart'
                    pic = 2

                elif rps_result[0]['rps'] == 'scissors':
                    text = 'face : dot'
                    pic = 3

                if pic == 1:
                    cv2.putText(image, text=text,
                                org=(rps_result[0]['org'][0], rps_result[0]['org'][1] + 70),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=3)
                    image = overlay_transparent(image, overlay, 300, 300, overlay_size=(100, 100))

                elif pic == 2:
                    cv2.putText(image, text=text,
                                org=(rps_result[0]['org'][0], rps_result[0]['org'][1] + 70),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=3)
                    image = overlay_transparent(image, overlay1, 300, 300, overlay_size=(100, 100))

                elif pic == 3:
                    cv2.putText(image, text=text,
                                org=(rps_result[0]['org'][0], rps_result[0]['org'][1] + 70),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=3)
                    image = overlay_transparent(image, overlay2, 300, 300, overlay_size=(100, 100))

    # Calculating the FPS
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    # Displaying FPS on the image
    cv2.putText(image, str(int(fps)) + " FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting image
    cv2.imshow("Facial and Hand Landmarks", image)

    # Enter key 'q' to break the loop
    if cv2.waitKey(1) == ord('q'):
        break

# When all the process is done
# Release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()


# # Code to access landmarks
# for landmark in mp_holistic.HandLandmark:
# 	print(landmark, landmark.value)
#
# print(mp_holistic.HandLandmark.WRIST.value)


