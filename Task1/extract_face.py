import cv2
import os
import shutil
import mediapipe as mp

mp_detect = mp.solutions.face_detection

# Input : image
# Output : 
# 1. Image with bounding box
# 2. Extracted bounding box area

def Img2Face(image):

    # Face detection
    with mp_detect.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector : 
        outputs = detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_bbox = image.copy()

        if outputs.detections is None :
            return
        
        for detection in outputs.detections :

            # Get bounding box
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            y = int(bbox.ymin * ih)
            x = int(bbox.xmin * iw)
            h = int(bbox.height * ih)
            w = int(bbox.width * iw)

            # Get Image with bounding box
            cv2.rectangle(image_bbox, (x, y), (x + w, y + h), (0, 0, 255), 5)
            
            # Get Face clip ; Bounding box area
            clip = image[y:y+h, x:x+w]

        return image_bbox, clip

def resize_height(image, height):
    image = image.copy()
    return cv2.resize(image, (height, int(image.shape[0] * (height / image.shape[1]))))

# Input : Directory of the extracted frames

def extract_face(INPATH, OUTPATH=""):

    if OUTPATH == "auto":
        OUTPATH = os.path.dirname(os.path.realpath(INPATH))

    # Create Directory for bbox, face clip
    bbox_DIR = os.path.join(OUTPATH,"BBoxs")
    clip_DIR = os.path.join(OUTPATH, "FaceClips")
    if os.path.isdir(bbox_DIR):
            shutil.rmtree(bbox_DIR)
    if os.path.isdir(clip_DIR):
            shutil.rmtree(clip_DIR)
    os.mkdir(bbox_DIR)
    os.mkdir(clip_DIR)

    # Windows
    cv2.namedWindow('BBox', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('BBox', 400, 200)
    cv2.namedWindow('Face Clip', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Face Clip', 400, 300)

    if not os.path.isfile(INPATH):
        image_list = os.listdir(INPATH)
        
        frame_count = 0

        for img_name in image_list :
            IMG_PATH = os.path.join(INPATH, img_name)
            image = cv2.imread(IMG_PATH)

            if image is None :
                print("Cannot read image file.")
                break

            if Img2Face(image) is None :
                continue
            
            bbox_image, clip = Img2Face(image)
            bbox_image = resize_height(bbox_image, 100)

            frame_count += 1
            BBOX_PATH = os.path.join(bbox_DIR, "bbox_{:03d}.jpg".format(frame_count))
            FACE_PATH = os.path.join(clip_DIR, "clip_{:03d}.jpg".format(frame_count))

            cv2.imshow('BBox', bbox_image)
            cv2.imshow('Face Clip', clip)
            cv2.imwrite(BBOX_PATH, bbox_image)
            cv2.imwrite(FACE_PATH, clip)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


