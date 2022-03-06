import cv2
import mediapipe as mp
import os
# pip install mediapipe

# example use: python3 demo_mediapipe_image.py

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

input_img_folder = "/media/boom/HDD/FanBu/Stuff/PhD/research/HP_Lab/HandPose/ROSbag/20220222/lowLightLidar1p/color"
# normalToF1p: 1645564599.865928
# normalStereo1p: 1645564600.177692
# normalLidar1p: 1645564034.126845

# lowLightToF1p: 1645564714.578840
# lowLightStereo1p: 1645564714.572248
# lowLightLidar1p: 1645564185.588453
input_img_name = "1645564185.588453.png"
input_img_path = os.path.join(input_img_folder, input_img_name)

result_img_name = "result_"+input_img_name

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    frame = cv2.imread(input_img_path)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_result = hands.process(frame)

    # Draw the hand annotations on the image.
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if frame_result.multi_hand_landmarks:
        for hand_landmarks in frame_result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

    # Using cv2.imwrite() method
    # Saving the image
    cv2.imwrite(result_img_name, frame)
