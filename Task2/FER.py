import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import matplotlib.pyplot as plt
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models.PosterV2_7cls import *
from models.PosterV2_8cls import *
from util import RecorderMeter, RecorderMeter1
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import mediapipe as mp

mp_detect = mp.solutions.face_detection

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]
)


def FER (traindata, VIDPATH) :

    if traindata == 'RAF-DB' or traindata == 'CAER-S' :
        model = pyramid_trans_expr2(img_size=224, num_classes=7)
        EMOTIONS = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Anger', 'Neutral']

    elif traindata == 'AffectNet-7' :
        model = pyramid_trans_expr2(img_size=224, num_classes=7)
        EMOTIONS = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger']

    elif traindata == 'AffectNet-8' :
        model = pyramid_trans_expr2(img_size=224, num_classes=8)
        EMOTIONS = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']

    else :
        print("No pretrained model selected")

    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    model_path = './checkpoint/' + traindata + '.pth'

    if os.path.isfile(model_path):

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])

        print("Loaded model from '{}'".format(model_path))

    else:
        print("No checkpoint found at '{}'".format(model_path))
        sys.exit(1)


    video = cv2.VideoCapture(VIDPATH)

    if not video.isOpened():
        print("Fail to open video file.")
        exit()

    cv2.namedWindow('FER', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('FER', 600, 400)


    while True :
        
        ret, frame = video.read()

        if not ret:
            print("Failed to read a frame from the video.")
            break

        with mp_detect.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector : 
            outputs = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_annot = frame.copy()

        for detection in outputs.detections :

            # Get bounding box
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            y = int(bbox.ymin * ih)
            x = int(bbox.xmin * iw)
            h = int(bbox.height * ih)
            w = int(bbox.width * iw)

            cv2.rectangle(image_annot, (x, y), (x + w, y + h), (0, 0, 255), 5)
            
            # Get Face clip ; Bounding box area
            face = frame[y:y+h, x:x+w]

            # Predict emotion for face area
            face = transform(image=face)['image']

            face = face.cuda()
            pred = model(face.unsqueeze(0)).view(-1)
            prob = nn.functional.softmax(pred) # To probability

            # Result annotation
            x = 10
            y = 30
            for emotion, p in zip(EMOTIONS, prob):
                text = f"{emotion}: {p:.2f}"
                cv2.putText(image_annot, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                y += 40

            # top1 = EMOTIONS[torch.argmax(prob).item()]
            # cv2.putText(image_annot, top1, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow('FER', image_annot)


        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
