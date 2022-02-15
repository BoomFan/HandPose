import cv2
import mediapipe as mp
import time
# pip install mediapipe


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Read webcam
cap = cv2.VideoCapture(0)

# Read video file
cap = cv2.VideoCapture(
    '/media/boom/HDD/FanBu/Stuff/PhD/research/HP_Lab/HandPose/ROSbag/20220128_143907/Yezhi.mp4')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    'Yezhi_mediapipe.mp4', fourcc, 30.0, (1280, 720))

current_fps = 0
fps_period = 5
frame_cnt = 0
time_passed = 0
# font
font = cv2.FONT_HERSHEY_SIMPLEX
# org
org = (50, 50)
# fontScale
fontScale = 1
# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 2

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    while(cap.isOpened()):
        success, frame = cap.read()
        if success is True:

            # process image
            frame_cnt += 1
            # Start time
            start = time.time()

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_frame = hands.process(frame)

            # End time
            end = time.time()
            # Time elapsed
            seconds = end - start
            time_passed += seconds
            # print(f"Time taken : {seconds} seconds")
            # Draw FPS on image
            if frame_cnt % fps_period == 0:
                # Calculate frames per second
                current_fps = frame_cnt / time_passed
                frame_cnt = 0
                time_passed = 0

            # Draw the hand annotations on the image.
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if result_frame.multi_hand_landmarks:
                for hand_landmarks in result_frame.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            # Using cv2.putText() method
            fps_string = f"FPS: {current_fps:.2f}"
            image = cv2.putText(frame, fps_string, org, font,
                                fontScale, color, thickness, cv2.LINE_AA)

            # write the result frame
            out.write(frame)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
