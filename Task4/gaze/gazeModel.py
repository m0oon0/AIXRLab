import torch
import cv2
from torchvision.transforms import ToTensor
import numpy as np
from gaze.model import Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ToTensor = ToTensor()

class GazeModel :
    def __init__(self):
              
        self.model = Model()
        self.model = self.model.to(DEVICE)

        # Load weights
        state_dict = torch.load('gaze/pretrained/GazeTR.pt')
        self.model.load_state_dict(state_dict)

        self.model.eval()

    def predict(self, face):

        # Resize & transform to Tensor (1x3x224x224)
        face = cv2.resize(face, (224, 224))
        face = ToTensor(face).unsqueeze(0).cuda()
        
        # To : model input form
        input_ = {'face': face}

        # Get model output (2 Dim)
        output = self.model(input_).detach().cpu().numpy().reshape(-1)
        gaze = [output[0], output[1]]

        return gaze