from flask import Flask, Response, render_template, jsonify, send_file
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vision.face_tracking import track_face_movement
from question_prompt.generate_question import get_question
import cv2
from flask_socketio import SocketIO, emit

# import requests
# import json


app            = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

camera         = cv2.VideoCapture(0)
# camera = cv2.VideoCapture("D:/Study/UIUC/Projects/head_nod_recognition/src/data/webcam_input.mp4")
current_question = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start():
    return Response(track_face_movement(camera,current_question,socketio), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/question')
def question():
    global current_question
    current_question = get_question()
    return {'question': current_question}

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('question', {'question': current_question})

if __name__ == '__main__':
    socketio.run(app, debug=True)