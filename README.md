# Cell_Classification_Models_and_GUI_Using_YOLO_and_PyQt5
 Cell classification models were trained using YOLOv8 and a GUI was built using PyQt5.
 Requirements:
 import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QFileDialog, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QScrollArea, QMessageBox, QComboBox
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt
from ultralytics import YOLO
import cv2
from collections import Counter
 
 
