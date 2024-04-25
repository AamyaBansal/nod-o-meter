"""import cv2
import numpy as np
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vision.utils import find_face, optical_flow_keypoint, current_frame_response, get_coords, axes_movements, define_gesture
from logs.logger import log_interaction

def track_face_movement(camera,current_question,socketio,face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
):
    face_found = False
    x_movement = 0
    y_movement = 0
    FRAME_RATE = 30
    FINAL_THRESHOLD = 10
    track_cur_gesture = FRAME_RATE
    # gesture_sequence = []
    flag_logged_first_response = False
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

        # DEBUG
        # print(f'prev: {coord_prev} vs cur: {coord_cur}')
        # print(f'x_movement {x_movement}, y movement {y_movement}, gesture: {gesture}')

        if gesture and (track_cur_gesture > 0):
            response = current_frame_response(coord_prev, coord_cur, face_rect, gesture, frame = frame_color, flag='correct')

            track_cur_gesture -= 1
            yield response
        
        if track_cur_gesture == 0:
            x_movement = 0
            y_movement = 0
            track_cur_gesture = FRAME_RATE
            flag_logged_first_response = False
        
        if (not flag_logged_first_response) and gesture != 'Try again': # FixMe - logic 
            log_interaction(current_question, gesture,socketio)
            flag_logged_first_response = True

        prev_face_coord = cur_face_coord
        old_gray = frame_gray.copy()
"""
import cv2
import numpy as np
import sys
import os
import argparse
from argparse import ArgumentParser
from vision.utils import find_face, optical_flow_keypoint, current_frame_response, get_coords, axes_movements, define_gesture
from logs.logger import log_interaction

from face_detection import FaceDetector
from mark_detection import MarkDetector
from pose_estimation import PoseEstimator
from utils import refine

print(__doc__)
print("OpenCV version: {}".format(cv2.__version__))

def track_face_movement(camera, current_question, socketio, face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')):
    face_found = False
    x_movement = 0
    y_movement = 0
    FRAME_RATE = 30
    FINAL_THRESHOLD = 10
    track_cur_gesture = FRAME_RATE
    # gesture_sequence = []
    flag_logged_first_response = False
    print('Running track face')
    
    # Initialize the video source from webcam.
    cap = cv2.VideoCapture(camera)
    
    # Get the frame size. This will be used by the following detectors.
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup a face detector to detect human faces.
    face_detector = FaceDetector("assets/face_detector.onnx")

    # Setup a mark detector to detect landmarks.
    mark_detector = MarkDetector("assets/face_landmarks.onnx")

    # Setup a pose estimator to solve pose.
    pose_estimator = PoseEstimator(frame_width, frame_height)

    # Measure the performance with a tick meter.
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

        # Step 1: Get faces from current frame.
        faces, _ = face_detector.detect(frame, 0.7)

        # Any valid face found?
        if len(faces) > 0:
            tm.start()

            # Step 2: Detect landmarks. Crop and feed the face area into the
            # mark detector. Note only the first face will be used for
            # demonstration.
            face = refine(faces, frame_width, frame_height, 0.15)[0]
            x1, y1, x2, y2 = face[:4].astype(int)
            patch = frame[y1:y2, x1:x2]

            # Run the mark detection.
            marks = mark_detector.detect([patch])[0].reshape([68, 2])

            # Convert the locations from local face area to the global image.
            marks *= (x2 - x1)
            marks[:, 0] += x1
            marks[:, 1] += y1

            # Step 3: Try pose estimation with 68 points.
            pose = pose_estimator.solve(marks)

            tm.stop()

            # All done. The best way to show the result would be drawing the
            # pose on the frame in realtime.

            # Do you want to see the pose annotation?
            pose_estimator.visualize(frame, pose, color=(0, 255, 0))

            # Do you want to see the axes?
            # pose_estimator.draw_axes(frame, pose)

            # Do you want to see the marks?
            # mark_detector.visualize(frame, marks, color=(0, 255, 0))

            # Do you want to see the face bounding boxes?
            # face_detector.visualize(frame, faces)

        # Draw the FPS on screen.
        cv2.rectangle(frame, (0, 0), (90, 30), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, f"FPS: {tm.getFPS():.0f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        # Show preview.
        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) == 27:
            break

if __name__ == '__main__':
    # Set the webcam index
    args = argparse.Namespace(cam=0)
    track_face_movement(args.cam, current_question=None, socketio=None)
