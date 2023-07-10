import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QMainWindow, QVBoxLayout, QTextEdit

import cv2
from extract_frame import Video2Imgs
from extract_face import extract_face

class GUIWindow(QMainWindow):

        def __init__(self):
                super().__init__()
                self.initUI()

        def initUI(self):
                self.setWindowTitle('Task 1')
                self.move(300, 300)
                self.resize(500, 500)
                self.show()

                self.button1 = QPushButton("Frame Extraction (Live)", self)
                self.button1.setGeometry(50, 50, 100, 30)
                self.button1.clicked.connect(lambda: self.extract_frames(live=True))

                self.button2 = QPushButton("Frame Extraction (Video)", self)
                self.button2.setGeometry(50, 60, 100, 30)
                self.button2.clicked.connect(self.extract_frames)

                self.button3 = QPushButton("Face Extraction", self)
                self.button3.setGeometry(50, 70, 100, 30)
                self.button3.clicked.connect(self.extract_faces)

                layout = QVBoxLayout()
                layout.addWidget(self.button1)
                layout.addWidget(self.button2)
                layout.addWidget(self.button3)

                self.text_output = QTextEdit()
                layout.addWidget(self.text_output)

                central_widget = QWidget(self)
                central_widget.setLayout(layout)
                self.setCentralWidget(central_widget)
        
        def extract_frames(self, live=False):
                options = QFileDialog.Options()
                options |= QFileDialog.DontUseNativeDialog

                # Select input video file
                if live==False:
                        VID_PATH, _ = QFileDialog.getOpenFileName(self, "Select Input Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv)", options=options)
                        if not VID_PATH:
                                return

                # Select output directory
                OUTPATH = QFileDialog.getExistingDirectory(self, "Select Output Directory", options=options)
                if not OUTPATH:
                        OUTPATH = "auto"
                
                if live == True :
                        frame_num = Video2Imgs(INPATH="camera", OUTPATH=OUTPATH)
                else :
                        frame_num = Video2Imgs(INPATH=VID_PATH, OUTPATH=OUTPATH)
                
                self.text_output.append(f"{frame_num} Frames extracted")

        def extract_faces(self):
                options = QFileDialog.Options()

                # Select input 
                FRAMES_PATH = QFileDialog.getExistingDirectory(self, "Select Input directory", "")
                if not FRAMES_PATH:
                        return
                
                # Select output directory
                OUTPATH = QFileDialog.getExistingDirectory(self, "Select Output Directory", options=options)
                if not OUTPATH:
                        OUTPATH = "auto"

                extract_face(INPATH=FRAMES_PATH, OUTPATH=OUTPATH)
                

if __name__ == '__main__':
   app = QApplication(sys.argv)
   window = GUIWindow()
   window.show()

   sys.exit(app.exec_())