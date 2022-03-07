import cv2
import mediapipe as mp
import os

# example use: python3 generate_depth_bboxes.py


def landmark_2_x_y_depth(hand_landmarks, depth_timestamp_str, depth_img_path, DEBUG_MODE=False):
    depth_frame = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)
    landmark_num = len(hand_landmarks.landmark)
    mark_x_avr = 0
    mark_y_avr = 0
    mark_depth_avr = 0
    valid_num = 0
    for landmark_idx in range(landmark_num):
        mark_x_normed = hand_landmarks.landmark[landmark_idx].x
        mark_y_normed = hand_landmarks.landmark[landmark_idx].y
        mark_z_normed = hand_landmarks.landmark[landmark_idx].z
        # z value has its 3D meaning, we just not use it here.
        # this website could help https://github.com/google/mediapipe/issues/742
        mark_x = int(color_frame_width*mark_x_normed)
        mark_x = min(max(mark_x, 0), color_frame_width-1)
        mark_y = int(color_frame_height*mark_y_normed)
        mark_y = min(max(mark_y, 0), color_frame_height-1)
        mark_depth = depth_frame[mark_y, mark_x]
        if mark_depth > 0 and mark_depth < 2000:
            mark_x_avr += mark_x
            mark_y_avr += mark_y
            mark_depth_avr += mark_depth
            valid_num += 1

        if DEBUG_MODE:
            # print("hand_landmarks = ", hand_landmarks)
            print("landmark_num = ", landmark_num)
            print("mark_x = ", mark_x)
            print("mark_y = ", mark_y)
            print(
                f"depth_frame[{mark_y}, {mark_x}] = {depth_frame[mark_y, mark_x]}")
    mark_x_avr = int(mark_x_avr/valid_num)
    mark_x_avr = min(max(mark_x_avr, 0),
                     color_frame_width-1)
    mark_y_avr = int(mark_y_avr/valid_num)
    mark_y_avr = min(max(mark_y_avr, 0),
                     color_frame_height-1)
    mark_depth_avr = int(mark_depth_avr/valid_num)
    # Detect one hand, save the result bbox in a txt file.
    # img_name    x_in_width    y_in_height    depth_in_mm
    landmark_str = depth_timestamp_str+".png"+"    " + \
        str(mark_x_avr) + "    "+str(mark_y_avr) + \
        "    "+str(mark_depth_avr)+"\n"
    return landmark_str


def landmark_2_bbox(hand_landmarks, depth_timestamp_str, bbox_extension):
    landmark_num = len(hand_landmarks.landmark)
    mark_x_list = []
    mark_y_list = []
    for landmark_idx in range(landmark_num):
        mark_x_normed = hand_landmarks.landmark[landmark_idx].x
        mark_y_normed = hand_landmarks.landmark[landmark_idx].y
        mark_z_normed = hand_landmarks.landmark[landmark_idx].z
        # z value has its 3D meaning, we just not use it here.
        # this website could help https://github.com/google/mediapipe/issues/742
        mark_x = int(color_frame_width*mark_x_normed)
        mark_x = min(max(mark_x, 0), color_frame_width-1)
        mark_x_list.append(mark_x)
        mark_y = int(color_frame_height*mark_y_normed)
        mark_y = min(max(mark_y, 0), color_frame_height-1)
        mark_y_list.append(mark_y)

    # Detect one hand, save the result bbox in a txt file.
    # img_name    x_min    y_min    x_max    y_max
    x_min = max(0, min(mark_x_list)-bbox_extension)
    y_min = max(0, min(mark_y_list)-bbox_extension)
    x_max = min(color_frame_width-1, max(mark_x_list)+bbox_extension)
    y_max = min(color_frame_height-1, max(mark_y_list)+bbox_extension)
    landmark_str = depth_timestamp_str+".png"+"    " + \
        str(x_min) + "    "+str(y_min) + "    " + \
        str(x_max)+"    "+str(y_max)+"\n"
    return landmark_str


if __name__ == '__main__':
    # ---------------Initialization------------------
    DEBUG_MODE = False

    bbox_extension = 100
    input_dataset_folder = "/media/boom/HDD/FanBu/Stuff/PhD/research/HP_Lab/HandPose/ROSbag/20220222/normalToF1p"
    input_color_img_folder = os.path.join(input_dataset_folder, "color")
    input_depth_img_folder = os.path.join(input_dataset_folder, "depth")
    result_txt_path = os.path.join(input_dataset_folder, "mdeiapipe_bbox.txt")

    # ---------------Codes start------------------
    file_object = open(result_txt_path, 'w')
    file_object.write('')
    file_object.close()

    # Get the color image names (timestamps)
    color_images_path = os.listdir(input_color_img_folder)
    color_timestamp_list = []
    for file_path in color_images_path:
        file_name = os.path.basename(file_path)
        timestamp = os.path.splitext(file_name)[0]
        color_timestamp_list.append(timestamp)

    # Sort the color image timestamps
    color_timestamp_list.sort()
    print(f"There are {len(color_timestamp_list)} color images.")

    # Get the depth image names (timestamps)
    depth_images_path = os.listdir(input_depth_img_folder)
    depth_timestamp_list = []
    for file_path in depth_images_path:
        file_name = os.path.basename(file_path)
        timestamp = os.path.splitext(file_name)[0]
        depth_timestamp_list.append(timestamp)

    # Sort the depth image timestamps
    depth_timestamp_list.sort()
    print(f"There are {len(depth_timestamp_list)} depth images.")

    # Initialize Mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # For each depth image,
    # run MediaPipe handpose detection on its previous closest color image
    color_ptr = 0
    depth_ptr = 0
    last_run_color_ptr = -1
    last_skeleton_result = []
    color_timestamp_str = color_timestamp_list[color_ptr]
    depth_timestamp_str = depth_timestamp_list[depth_ptr]

    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        file_object = open(result_txt_path, 'a')
        while depth_ptr < len(depth_timestamp_list):
            color_timestamp_str = color_timestamp_list[color_ptr]
            depth_timestamp_str = depth_timestamp_list[depth_ptr]
            color_timestamp = float(color_timestamp_str)
            depth_timestamp = float(depth_timestamp_str)
            # if depth time is behind color time, move depth
            if color_timestamp > depth_timestamp:
                depth_ptr += 1
                continue
            # if depth time is after or equal color time, make sure color time is the closest
            if color_timestamp < depth_timestamp:
                if color_ptr < len(color_timestamp_list)-1:
                    color_timestamp_next = float(
                        color_timestamp_list[color_ptr+1])
                    if color_timestamp_next <= depth_timestamp:
                        color_ptr += 1
                        continue

            # Now we find the closest color image, check if this image has been run
            if last_run_color_ptr != color_ptr:
                # we can use color image detection to estimate a rough 3D bbox
                color_img_path = os.path.join(
                    input_color_img_folder, color_timestamp_str+".png")
                color_frame = cv2.imread(color_img_path)
                color_frame_height = color_frame.shape[0]
                color_frame_width = color_frame.shape[1]
                depth_img_path = os.path.join(
                    input_depth_img_folder, depth_timestamp_list[depth_ptr]+".png")
                depth_frame = cv2.imread(depth_img_path)
                depth_frame_height = depth_frame.shape[0]
                depth_frame_width = depth_frame.shape[1]
                if color_frame_height != depth_frame_height:
                    assert False, "Please align two images!"
                if color_frame_width != depth_frame_width:
                    assert False, "Please align two images!"

                print(
                    f"color_ptr {color_ptr} match with depth_ptr {depth_ptr}")
                if DEBUG_MODE:
                    print("color_frame.shape = ", color_frame.shape)
                    print("depth_frame.shape = ", depth_frame.shape)
                    print(
                        f"color time {color_timestamp_str} match with depth time {depth_timestamp_str}")

                color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
                frame_result = hands.process(color_frame)

                if frame_result.multi_hand_landmarks:
                    for hand_landmarks in frame_result.multi_hand_landmarks:
                        # landmark_str = landmark_2_x_y_depth(
                        #     hand_landmarks, depth_timestamp_str, depth_img_path)
                        landmark_str = landmark_2_bbox(
                            hand_landmarks, depth_timestamp_str, bbox_extension)
                        file_object.write(landmark_str)

                        if DEBUG_MODE:
                            depth_frame_debug = depth_frame.copy()
                            color_frame_debug = cv2.cvtColor(
                                color_frame, cv2.COLOR_RGB2BGR)
                            depth_frame_debug.flags.writeable = True

                            mp_drawing.draw_landmarks(
                                depth_frame_debug,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style())
                            mp_drawing.draw_landmarks(
                                color_frame_debug,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style())
                            cv2.imshow("depth_frame_debug", depth_frame_debug)
                            cv2.imshow("color_frame_debug", color_frame_debug)
                            cv2.waitKey(0)

                    last_run_color_ptr = color_ptr
            else:
                skeleton_result = last_skeleton_result

            # Move both image pointer forward
            color_ptr += 1
            depth_ptr += 1

        file_object.close()
