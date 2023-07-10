import os
import cv2
import shutil

# Input : Live camera or Video file path
# Output :
# Save frames
# Return # of frames

def Video2Imgs(INPATH, OUTPATH, savefps=15):

    if INPATH == "camera" :
        video = cv2.VideoCapture(0)
    else :
        video = cv2.VideoCapture(INPATH)

    if OUTPATH == "auto":
        OUTPATH = os.path.join(os.path.dirname(os.path.realpath(INPATH)), "Frames")
        if os.path.isdir(OUTPATH):
            shutil.rmtree(OUTPATH)
        os.mkdir(OUTPATH)

    # Calculate the frame interval for saving
    save_fps = savefps
    source_fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(source_fps / save_fps))

    if not video.isOpened():
        print("Fail to open video file.")
        exit()

    # Window
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Frame', 800, 600)
    cv2.moveWindow('Frame', 100, 100)

    frame_count = 0
    frame_index = 0

    while True :
        ret, frame = video.read()

        if not ret:
            print("Failed to read a frame from the video.")
            break

        if frame_count % frame_interval == 0 :
            IMG_PATH = os.path.join(OUTPATH, "frame_{:03d}.jpg".format(frame_index))
            cv2.imwrite(IMG_PATH, frame)
            cv2.imshow('Frame', frame)
            frame_index += 1

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video.release()
    cv2.destroyAllWindows()

    return frame_index