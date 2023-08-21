from flask import Flask, render_template, request, jsonify, Response
from videoModel import VideoModel
from audioModel import AudioModel
import cv2
import tempfile
import base64
from flask_cors import CORS
import json
import pandas as pd
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

INPUTDIR = ""

PRONUN_LIST = ['아', '에', '이', '오', '우', '으', '어', 'ㅁㅂㅍ', '기타']
EMOTION_LIST = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Anger', 'Neutral']
GLOBAL_HEATMAP = pd.DataFrame(0, index=range(0, len(EMOTION_LIST)), columns=PRONUN_LIST)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_analysis')
def start_analysis(data):

    inputVideo = data['inputVideo']
    emotionOpt = data['emotionOpt']
    gazeOpt = data['gazeOpt']
    voiceOpt = data['voiceOpt']

    video_path = INPUTDIR + inputVideo

    socketio.start_background_task(target=process_video, video_path=video_path, emotionOpt=emotionOpt, gazeOpt=gazeOpt, voiceOpt=voiceOpt)


def process_video(video_path, emotionOpt, gazeOpt, voiceOpt):

    model = VideoModel()

    cap = cv2.VideoCapture(video_path)

    # Store emotion per frame
    video_emotions = []
    frame_num = 0
    
    while True:
        ret, frame = cap.read()
        frame_num += 1

        if not ret :
            print("End of video")
            break

        if 'change' in gazeOpt:
            try :
                annotated_frame, emotion, gaze = model.process(frame, frame_num, emotionOpt, gazeOpt)
                video_emotions.append(emotion)
            
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                annotated_frame = buffer.tobytes()

                socketio.emit('frame_update', {'frame': annotated_frame, 'gaze': gaze, 'frame_num': frame_num})

            except :
                pass

        else :
            try :
                annotated_frame, emotion = model.process(frame, frame_num, emotionOpt, gazeOpt)
                video_emotions.append(emotion)

                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                annotated_frame = buffer.tobytes()

                socketio.emit('frame_update', {'frame': annotated_frame})
            except :
                pass

    if voiceOpt == "stt" :
        process_audio(video_path, video_emotions)


def process_audio(video_path, video_emotions):
    
    global GLOBAL_HEATMAP

    model = AudioModel(video_path)

    # emotion color for each syllable
    text, color = model.get_speech_color(video_emotions)

    # emotion - pronunciation heatmap
    heatmap_cv2, heatmap_df = model.get_heatmap(GLOBAL_HEATMAP)
    GLOBAL_HEATMAP = heatmap_df
    ret, buffer = cv2.imencode('.jpg', heatmap_cv2)
    heatmap = buffer.tobytes()

    socketio.emit('stt_update', {'text': text, 'color': color, 'heatmap': heatmap})


if __name__ == '__main__':
    app.run(debug=True)
    socketio.run(app)