import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import mediapipe as mp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from emotion.emotionModel import EmotionModel
from gaze.gazeModel import GazeModel

class VideoModel():
    def __init__(self):

        mp_detect = mp.solutions.face_detection
        self.detector = mp_detect.FaceDetection(model_selection=1, min_detection_confidence=0.5)

        self.emotionModel = EmotionModel()
        self.EMOTIONS = self.emotionModel.EMOTIONS
        self.gazeModel = GazeModel()

    def gazeto3d(self, gaze):
        assert len(gaze) == 2
        gaze_gt = np.zeros([3])
        gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
        gaze_gt[1] = -np.sin(gaze[1])
        gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
        return gaze_gt
    
    def process(self, frame, frame_num, emotionOpt="", gazeOpt=[]):

        ### Face Detection
        face_detections = self.detector.process(frame)
        image_annot = cv2.resize(frame.copy(), (480, 640))

        frameNo = "frame #{:03d}".format(frame_num)
        loc = (10, image_annot.shape[0] - 20)
        cv2.putText(image_annot, frameNo, loc, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        emotion_top1 = ""

        for detection in face_detections.detections :

            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            y = int(bbox.ymin * ih)
            x = int(bbox.xmin * iw)
            h = int(bbox.height * ih)
            w = int(bbox.width * iw)
            
            # Face area
            face = frame[y:y+h, x:x+w]

            ##################################################################################################
            ### Emotion recognition
            if emotionOpt != "" :

                # Predict emotion for face area
                probs = self.emotionModel.predict(face)

                emotion_prob = list(zip(self.EMOTIONS, probs))
                sorted_emotion_prob = sorted(emotion_prob, key=lambda x: x[1], reverse=True)

                # Result annotation
                x = 10
                y = 30

                if emotionOpt == 'all' :
                    n = len(self.EMOTIONS)
                elif emotionOpt == 'top3' :
                    n = 3
                elif emotionOpt == 'top1' :
                    n = 1
                    
                for emotion, p in sorted_emotion_prob[:n] :
                        # text = f"{emotion}: {p:.2f}"
                        text = f"{emotion}"
                        cv2.putText(image_annot, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        y += 40

                emotion_top1 = sorted_emotion_prob[0][0]

            ##################################################################################################
            ### Gaze tracking
            if gazeOpt != [] :
                # Get output from GazeTR model
                gaze = self.gazeModel.predict(face)

                height, width, _ = image_annot.shape

                # For Visualization
                if 'value' in gazeOpt:
                    origin = [width - 150, 50]

                    yaw = gaze[0]
                    pitch = gaze[1]

                    cv2.putText(image_annot, f"({yaw:.2f}, {pitch:.2f})", origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                elif 'vector' in gazeOpt:
                    origin = [width - 100, 100]
                    length = 50

                    # 2d to 3d
                    gaze_3d = self.gazeto3d(gaze)

                    gaze_3d = gaze/np.linalg.norm(gaze_3d) * length
                    gaze_3d = gaze_3d.astype(int)

                    cv2.arrowedLine(image_annot, origin, (origin[0] - gaze_3d[0], origin[1] + gaze_3d[1]), (0, 0, 255), 3)

                if 'change' in gazeOpt:
                    return image_annot, emotion_top1, [float(value) for value in gaze]

        return image_annot, emotion_top1