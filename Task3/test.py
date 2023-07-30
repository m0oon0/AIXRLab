import torch
from model import Model
import cv2
import mediapipe as mp
from torchvision.transforms import ToTensor
import numpy as np

mp_detect = mp.solutions.face_detection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ToTensor = ToTensor()

# Import model
model = Model()
model = model.to(DEVICE)

# Load weights
state_dict = torch.load('./pretrained/GazeTR.pt')
model.load_state_dict(state_dict)

model.eval()

# Input : Detected face area (any size)
# Output : Yaw, Pitch

def get_gaze_direction(face) :

    # Resize & transform to Tensor (1x3x224x224)
    face = cv2.resize(face, (224, 224))
    face = ToTensor(face).unsqueeze(0).cuda()
    
    # To : model input form
    input_ = {'face': face}

    # Get model output (2 Dim)
    gaze = model(input_).view(-1)

    yaw = gaze[0].item()
    pitch = gaze[1].item()

    return yaw, pitch


### Test function
# Input : Video path
# Output : Annotated image (window)

def test(VIDPATH):

    # Load video
    video = cv2.VideoCapture(VIDPATH)

    if not video.isOpened():
        print("Fail to open video file.")
        exit()

    cv2.namedWindow('Gaze', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Gaze', 600, 400)

    while True :
        
        ret, frame = video.read()

        # For Visualization
        image_annot = frame.copy()

        if not ret:
            print("Failed to read a frame from the video.")
            break

        with mp_detect.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector : 
            outputs = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        for detection in outputs.detections :

            # Get bounding box
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            y = int(bbox.ymin * ih)
            x = int(bbox.xmin * iw)
            h = int(bbox.height * ih)
            w = int(bbox.width * iw)
            
            # Get Face area
            face = frame[y:y+h, x:x+w]

            # Get output from GazeTR model
            yaw, pitch = get_gaze_direction(face)

            # For Visualization
            cv2.rectangle(image_annot, (x, y), (x + w, y + h), (0, 0, 255), 5)

            cv2.putText(image_annot, f"yaw {yaw:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
            cv2.putText(image_annot, f"pitch {pitch:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
            
        cv2.imshow('Gaze', image_annot)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
