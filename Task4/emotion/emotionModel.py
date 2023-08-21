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
from emotion.models.PosterV2_7cls import *
from emotion.models.PosterV2_8cls import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]
)

class EmotionModel:
    def __init__(self, traindata='RAF-DB'):

        if traindata == 'RAF-DB' or traindata == 'CAER-S' :
            self.model = pyramid_trans_expr2(img_size=224, num_classes=7)
            self.EMOTIONS = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Anger', 'Neutral']

        elif traindata == 'AffectNet-7' :
            self.model = pyramid_trans_expr2(img_size=224, num_classes=7)
            self.EMOTIONS = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger']

        elif traindata == 'AffectNet-8' :
            self.model = pyramid_trans_expr2(img_size=224, num_classes=8)
            self.EMOTIONS = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']

        else :
            print("No pretrained model selected")

        self.model = torch.nn.DataParallel(self.model).cuda()
        self.model.eval()

        model_path = 'emotion/checkpoint/' + traindata + '.pth'

        if os.path.isfile(model_path):

            # checkpoint = torch.load(model_path)
            self.model.load_state_dict(torch.load(model_path))

            print("Loaded model from '{}'".format(model_path))

        else:
            print("No checkpoint found at '{}'".format(model_path))
            sys.exit(1)

    def predict(self, face):

        # Predict emotion for face area
        face = transform(image=face)['image']

        face = face.cuda()
        pred = self.model(face.unsqueeze(0)).view(-1)
        prob = nn.functional.softmax(pred) # To probability

        return prob
