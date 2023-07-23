import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QMainWindow, QVBoxLayout, QTextEdit, QComboBox, QLabel

import cv2
from FER import FER
from util import RecorderMeter, RecorderMeter1


class GUIWindow(QMainWindow):

        def __init__(self):
                super().__init__()
                self.initUI()

        def initUI(self):
                self.setWindowTitle('TASK2')
                self.move(300, 300)
                self.resize(500, 500)
                self.show()

                # trained model selection
                self.selectbox1 = QComboBox()
                self.selectbox1.setGeometry(50, 50, 100, 30)
                self.selectbox1.addItem('RAF-DB')
                self.selectbox1.addItem('AffectNet-7')
                self.selectbox1.addItem('AffectNet-8')
                self.selectbox1.addItem('CAER-S')
                # self.selectbox1.currentIndexChanged.connect(self.on_combo_box_changed)

                # Select data path
                self.button1 = QPushButton("Select Video", self)
                self.button1.setGeometry(50, 60, 100, 30)
                self.button1.clicked.connect(self.select_video)

                self.button2 = QPushButton("Run FER", self)
                self.button2.setGeometry(50, 70, 100, 30)
                self.button2.clicked.connect(self.run_FER)

                layout = QVBoxLayout()
                layout.addWidget(self.selectbox1)
                layout.addWidget(self.button1)
                layout.addWidget(self.button2)

                central_widget = QWidget(self)
                central_widget.setLayout(layout)
                self.setCentralWidget(central_widget)


        def select_video(self):
                options = QFileDialog.Options()
                self.VIDPATH, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv)", options=options)
                        

        def run_FER(self):
                FER(traindata=self.selectbox1.currentText(), VIDPATH=self.VIDPATH)
                

if __name__ == '__main__':
   app = QApplication(sys.argv)
   window = GUIWindow()
   window.show()

   sys.exit(app.exec_())