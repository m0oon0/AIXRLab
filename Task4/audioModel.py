from moviepy.editor import VideoFileClip
import hgtk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from google.cloud import speech
import statistics

sns.set(font_scale=1.5, font="Malgun Gothic")

PRONUN_LIST = ['아', '에', '이', '오', '우', '으', '어', 'ㅁㅂㅍ', '기타']
EMOTION_LIST = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Anger', 'Neutral', ' ']
COLOR_LIST = ['orange', 'gray', 'green', 'pink', 'blue', 'red', 'black', ' ']

credentials_path = ""

class AudioModel :

    def __init__(self, video_path):

        self.video_path = video_path
        
    def get_text_pronun(self, text):

        text_pronun = [self.get_pronun(syllable) for syllable in text]
        return text_pronun
        
    def get_pronun(self, syllable):

        try : 
            choseong, jungseong, jongseong = hgtk.letter.decompose(syllable)

            if jongseong in ['ㅁ', 'ㅂ', 'ㅍ'] :
                return 'ㅁㅂㅍ'

            if jungseong in ['ㅏ', 'ㅑ'] :
                return '아'

            elif jungseong in ['ㅔ', 'ㅖ', 'ㅐ', 'ㅒ', 'ㅚ', 'ㅙ'] :
                return '에'

            elif jungseong in ['ㅣ', 'ㅢ', 'ㅟ'] :
                return '이'

            elif jungseong in ['ㅗ', 'ㅛ'] :
                return '오'

            elif jungseong in ['ㅜ', 'ㅠ'] :
                return '우'

            elif jungseong in ['ㅡ'] :
                return '으'

            elif jungseong in ['ㅓ', 'ㅕ'] :
                return '어'

            return '기타'
        
        except :
            return '기타'


    def get_syllable_frame(self):

        audio_path = './audio.wav'

        client = speech.SpeechClient.from_service_account_json(credentials_path)

        video = VideoFileClip(self.video_path)
        FPS = video.fps
        audio = video.audio
        audio.write_audiofile(audio_path, codec="pcm_s16le", ffmpeg_params=["-ac", "1"])

        with open(audio_path, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,
            language_code="ko-KR",
            enable_word_time_offsets=True,
        )

        response = client.recognize(config=config, audio=audio)

        SYLLABLE_FRAME = []

        for result in response.results:

            alternative = result.alternatives[0]

            for word_info in alternative.words:
                word = word_info.word

                start_frame = int(word_info.start_time.total_seconds() * FPS)
                end_frame = int(word_info.end_time.total_seconds() * FPS)

                num_divisions = len(word)

                frame_range = end_frame - start_frame + 1
                frames_per_division = frame_range // num_divisions

                for w in word :
                    start = start_frame
                    end = start_frame + frames_per_division

                    SYLLABLE_FRAME.append([w, start, end])
                    start_frame = end + 1

                SYLLABLE_FRAME.append([" "])

        return SYLLABLE_FRAME


    def get_speech_color(self, video_emotions):
        
        syllable_frame = self.get_syllable_frame()

        if len(video_emotions) < len(syllable_frame) :
            video_emotions.extend(['Neutral'] * (len(syllable_frame) - len(video_emotions)))

        self.syllable_list = []
        self.syllable_emotion_list = []

        for sf in syllable_frame : 

            if sf[0] == " " :
                self.syllable_list.append(" ")
                self.syllable_emotion_list.append(" ")

            else :

                emotions_per_syllable = video_emotions[sf[1] : sf[2] + 1]

                self.syllable_list.append(sf[0])
                self.syllable_emotion_list.append(statistics.mode(emotions_per_syllable))


        index = [EMOTION_LIST.index(emotion) for emotion in self.syllable_emotion_list]
        color = [COLOR_LIST[i] for i in index]
        
        return self.syllable_list, color


    def get_heatmap(self, heatmap_df):

        text = [_ for _ in self.syllable_list if _ != " "]
        pronuns = self.get_text_pronun(text)

        emotions = [_ for _ in self.syllable_emotion_list if _ != " "]

        for pronun, emotion in zip(pronuns, emotions) :
            heatmap_df.loc[EMOTION_LIST.index(emotion), pronun] += 1

        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_df, xticklabels=PRONUN_LIST, yticklabels=EMOTION_LIST, cmap='Blues', cbar=False, linewidths=.5)

        figure_canvas = plt.gcf().canvas
        figure_canvas.draw()
        heatmap_np = np.array(figure_canvas.renderer.buffer_rgba())

        heatmap_cv2 = cv2.cvtColor(heatmap_np, cv2.COLOR_RGBA2BGR)

        return heatmap_cv2, heatmap_df
