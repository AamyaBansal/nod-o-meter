import logging
import datetime


def log_interaction(question, gesture,socketio):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f'Question: {question} - Gesture: {gesture} - Timestamp: {timestamp}')
    socketio.emit('log', f'Question: {question} - Gesture: {gesture} - Timestamp: {timestamp}')
