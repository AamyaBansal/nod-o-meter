from flask import Flask, Response, render_template, jsonify, send_file
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vision.face_tracking import track_face_movement
from question_prompt.generate_question import get_question
import cv2
# import requests
# import json


app            = Flask(__name__)
camera         = cv2.VideoCapture(0)
# camera = cv2.VideoCapture("D:/Study/UIUC/Projects/head_nod_recognition/src/data/webcam_input.mp4")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start():
    return Response(track_face_movement(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/question')
def question():
    return {'question': get_question()}

if __name__ == '__main__':
    app.run(debug=True)
