"""import cv2
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
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')"""

import cv2
import numpy as np
import sys
import os
from argparse import ArgumentParser
from face_detection import FaceDetector
from mark_detection import MarkDetector
from pose_estimation import PoseEstimator
from utils import refine
import cv2
import numpy as np
import sys
import os
import argparse
from argparse import ArgumentParser
from vision.utils import find_face, optical_flow_keypoint, current_frame_response, get_coords, axes_movements, define_gesture
from logs.logger import log_interaction

print(__doc__)
print("OpenCV version: {}".format(cv2.__version__))

def get_coords(cur_face_coord):
    try:
        return int(cur_face_coord[0][0][0]), int(cur_face_coord[0][0][1])
    except:
        return int(cur_face_coord[0][0]), int(cur_face_coord[0][1])

def optical_flow_keypoint(old_gray, frame_gray, prev_face_coord):
    return cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, prev_face_coord, None, winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

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
            return 'No'
        elif y_movement > x_movement:
            return 'Yes'

def track_face_movement(camera, current_question, socketio, face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')):
    face_found = False
    x_movement = 0
    y_movement = 0
    FRAME_RATE = 30
    FINAL_THRESHOLD = 10
    track_cur_gesture = FRAME_RATE
    flag_logged_first_response = False
    print('Running track face')

    cap = cv2.VideoCapture(camera)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    face_detector = FaceDetector("C:\\Users\\DELL\\PycharmProjects\\pythonProject5\\head-pose-estimation\\assets\\face_detector.onnx")
    mark_detector = MarkDetector("C:\\Users\\DELL\\PycharmProjects\\pythonProject5\\head-pose-estimation\\assets\\face_landmarks.onnx")
    pose_estimator = PoseEstimator(frame_width, frame_height)

    tm = cv2.TickMeter()

    while True:
        ret, frame = cap.read()
        if not ret:
            print('broken')
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_color = frame.copy()
        face_rect = find_face(frame_gray, face_cascade)

        if (not face_found) and (face_rect is not None):
            x, y, w, h = face_rect
            face_center = x + w // 2, y + h // 3

            prev_face_coord = np.array([[face_center]], np.float32)
            old_gray = frame_gray.copy()
            face_found = True
            continue

        if face_rect is None:
            face_found = False
            yield current_frame_response(frame=frame_color, flag='incorrect')
            continue

        cur_face_coord, st, err = optical_flow_keypoint(old_gray, frame_gray, prev_face_coord)
        coord_prev, coord_cur = get_coords(prev_face_coord), get_coords(cur_face_coord)
        shift_x, shift_y = axes_movements(coord_prev, coord_cur)
        x_movement += shift_x
        y_movement += shift_y

        faces, _ = face_detector.detect(frame, 0.7)

        if len(faces) > 0:
            tm.start()
            face = refine(faces, frame_width, frame_height, 0.15)[0]
            x1, y1, x2, y2 = face[:4].astype(int)
            patch = frame[y1:y2, x1:x2]

            marks = mark_detector.detect([patch])[0].reshape([68, 2])
            marks *= (x2 - x1)
            marks[:, 0] += x1
            marks[:, 1] += y1

            pose = pose_estimator.solve(marks)

            tm.stop()

            pose_estimator.visualize(frame, pose, color=(0, 255, 0))

        cv2.rectangle(frame, (0, 0), (90, 30), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, f"FPS: {tm.getFPS():.0f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) == 27:
            break

if __name__ == '__main__':
    args = argparse.Namespace(cam=0)
    track_face_movement(args.cam, current_question=None, socketio=None)
