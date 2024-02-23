import cv2
import numpy as np

def get_coords(cur_face_coord):
    try:
        return int(cur_face_coord[0][0][0]), int(cur_face_coord[0][0][1])
    except:
        return int(cur_face_coord[0][0]), int(cur_face_coord[0][1])


def optical_flow_keypoint(old_gray, frame_gray, prev_face_coord):
    return cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, prev_face_coord, None, winSize=(15, 15), maxLevel=2,
                                                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
def find_face(frame_gray, face_cascade):
    faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
    if len(faces) > 0:
        return faces[0]
    return None

def axes_movements(coord_prev, coord_cur):
    x_axis_movement = abs(coord_prev[0] - coord_cur[0])
    y_axis_movement = abs(coord_prev[1] - coord_cur[1])

    return x_axis_movement, y_axis_movement

def define_gesture(x_movement, y_movement, gesture_threshold=30):

    if max(x_movement,y_movement) < gesture_threshold or abs(x_movement-y_movement) < gesture_threshold:
        return 'Try again'
    else:
        if x_movement >= y_movement:
            return  'No'
        elif y_movement > x_movement:
            return 'Yes'
        


def current_frame_response(coord_prev=None, coord_cur=None, face_rect=None, gesture=None, frame=None, flag='correct'):
    if flag == 'correct':
        x, y, w, h = face_rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if gesture != 'Try again':
            cv2.putText(frame, gesture, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        return (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    else:
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        return (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')