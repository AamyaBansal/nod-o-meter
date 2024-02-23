import cv2
import numpy as np
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vision.utils import find_face, optical_flow_keypoint, current_frame_response, get_coords, axes_movements, define_gesture

def track_face_movement(camera,face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
):
    face_found = False
    x_movement = 0
    y_movement = 0
    FRAME_RATE = 30
    track_cur_gesture = FRAME_RATE
    print('Running track face')
    while True:
        ret, frame = camera.read()
        # print('frame shape',frame)
        if not ret: 
            
            print('broken')
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_color = frame.copy()
        face_rect  = find_face(frame_gray, face_cascade)

        if (not face_found) and (face_rect is not None):
            x, y, w, h  = face_rect
            face_center = x + w // 2, y + h // 3

            prev_face_coord = np.array([[face_center]], np.float32)
            old_gray = frame_gray.copy()
            face_found = True
            continue

        if face_rect is None:
            face_found = False
            yield current_frame_response(frame = frame_color, flag='incorrect')
            continue

        cur_face_coord, st, err = optical_flow_keypoint(old_gray, frame_gray, prev_face_coord)
        coord_prev, coord_cur   = get_coords(prev_face_coord), get_coords(cur_face_coord)
        shift_x, shift_y  = axes_movements(coord_prev, coord_cur)
        x_movement += shift_x
        y_movement += shift_y

        gesture  = define_gesture(x_movement, y_movement)
        print(f'prev: {coord_prev} vs cur: {coord_cur}')
        print(f'x_movement {x_movement}, y movement {y_movement}, gesture: {gesture}')

        if gesture and (track_cur_gesture > 0):
            response = current_frame_response(coord_prev, coord_cur, face_rect, gesture, frame = frame_color, flag='correct')

            track_cur_gesture -= 1
            yield response
        
        if track_cur_gesture == 0:
            x_movement = 0
            y_movement = 0
            track_cur_gesture = FRAME_RATE

        prev_face_coord = cur_face_coord
        old_gray = frame_gray.copy()

        

    # cap.release()
