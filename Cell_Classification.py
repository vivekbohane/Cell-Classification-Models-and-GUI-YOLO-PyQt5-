
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QFileDialog, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QScrollArea, QMessageBox, QComboBox
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt
from ultralytics import YOLO
import cv2
from collections import Counter
 
 
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set window properties
        self.setWindowTitle('Cell Classification')
        self.setGeometry(100, 100, 1400, 700)
        
        # Create a QWidget for the title section
        title_widget = QWidget(self)
        title_layout = QHBoxLayout(title_widget)  # Horizontal layout for title and image
        title_layout.setContentsMargins(20, 0, 0, 0)  # Remove any extra margins

        # Create a QLabel for a custom title within the window
        title_label = QLabel("Cell Classification", self)
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(40)  # Set font size
        title_font.setBold(True)     # Make it bold if desired
        title_label.setFont(title_font)

        # Create a QLabel for the image
        image_label = QLabel(self)
        image_label.setFixedSize(240, 240)  # Set a fixed size for the image
        image_label.setAlignment(Qt.AlignLeft)
        image_label.setPixmap(QPixmap("cell_img.png").scaled(240, 240, Qt.KeepAspectRatio))

        # Create a QLabel for displaying model-specific information
        self.info_label = QLabel(self)
        self.info_label.setFixedSize(400, 240)  # Set fixed size for the info box
        self.info_label.setStyleSheet(
            "border: 2px solid black; font-size: 22px; padding: 10px; background-color: white;"
        )
        self.info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.info_label.setWordWrap(True)
        self.info_label.setText("Select a model to see information here.")  # Default text

        # Add the image and title to the horizontal layout
        title_layout.addWidget(image_label)
        title_layout.addWidget(title_label)
        # Add the info_label to the title layout beside the title and image
        title_layout.addWidget(self.info_label)

        # Define a dictionary containing information for each model
        self.model_info = {
            "YOLO 7_Cell_Class_Type": """7_Cell_Class_Type ( 7 Classes )
        - Basophil
        - Eosinophil
        - Lymphocyte
        - Monocyte
        - Neutrophil
        - Platelets
        - RBC
        """,
            "YOLO Malaria": """Malaria ( 2 Classes )
        - Infected
        - Notinfected
        """,
            "YOLO WBC_Subtypes": """WBC_Subtypes ( 4 Classes )
        - Neutrophil
        - Lymphocyte
        - Monocyte
        - Eosinophil
        """,
            "YOLO Leukemia_Cancer": """Leukemia_Cancer ( 4 Classes )
        - Benign
        - Early
        - Pre
        - Pro
        """,
        }

        
        # Load the YOLO model
        self.model = YOLO('7_cell_class_type_yolov8.pt')
        # model = torch.load('best_swetha.pt')


        # Create a QWidget to hold the layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # Create the main layout for the central widget
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Add the title label to the main layout
        # Add the title widget to the main layout
        self.main_layout.addWidget(title_widget)
        # Create a horizontal layout to hold two vertical layouts
        self.horizontal_layout = QHBoxLayout()
        self.horizontal_layout.setAlignment(Qt.AlignLeft)

        # Create two vertical layouts for images and detected objects
        self.image_layout = QVBoxLayout()
        self.image_layout.setAlignment(Qt.AlignTop)
        self.image_layout_2 = QVBoxLayout()
        self.text_layout = QVBoxLayout()
        
        # Create a scroll area to hold the horizontal layout
        self.scroll_area = QScrollArea(self)
        self.scroll_area_widget = QWidget(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.scroll_area_widget)
        self.scroll_area_widget.setLayout(self.horizontal_layout)

        # Add the vertical layouts to the horizontal layout
        self.horizontal_layout.addLayout(self.image_layout)
        self.horizontal_layout.addLayout(self.image_layout_2)
        self.horizontal_layout.addLayout(self.text_layout)

        # Create a button layout
        self.button_layout = QHBoxLayout()

        # Create buttons with larger size
        self.load_image_button = QPushButton('Load Images', self)
        self.load_image_button.clicked.connect(self.load_images)
        self.load_image_button.setFixedSize(200, 40)
        self.load_image_button.setStyleSheet(
            "background-color: lightblue; color: black; font-size: 23px; font-weight: bold; border: 2px solid black;"
        )

        # Create a combo box for model selection
        self.model_combo_box = QComboBox(self)
        self.model_combo_box.addItem("Select Model")  # Add placeholder text
        self.model_combo_box.addItem("YOLO 7_Cell_Class_Type")
        self.model_combo_box.addItem("YOLO Malaria")
        self.model_combo_box.addItem("YOLO WBC_Subtypes")
        self.model_combo_box.addItem("YOLO Leukemia_Cancer")
        self.model_combo_box.setFixedSize(400, 40)
        self.model_combo_box.setStyleSheet(
            "QComboBox { "
            "background-color: lightcoral; "
            "color: black; "
            "font-size: 23px; "
            "border: 2px solid black; "
            "padding: 0px 15px; "  # Adds 5px padding on the left and right
            "} "
            "QComboBox QAbstractItemView { "
            "background-color: white; "
            "color: black; "
            "}"
        )
        self.model_combo_box.setCurrentIndex(0)  # Set "Select Model" as default
        self.model_combo_box.currentTextChanged.connect(self.handle_model_selection)
        self.model_combo_box.currentTextChanged.connect(self.select_model_from_combo)
        # Connect the combo box's text change signal to the update function
        self.model_combo_box.currentTextChanged.connect(self.update_info_label)
        
        # Create the Detect Objects button
        self.detect_button = QPushButton('Detect Objects', self)
        self.detect_button.setEnabled(False)  # Initially disable the button
        self.detect_button.clicked.connect(self.detect_objects)
        self.detect_button.setFixedSize(200, 40)
        self.detect_button.setStyleSheet(
            "background-color: lightgreen; color: black; font-size: 23px; font-weight: bold; border: 2px solid black;"
        )


        # Add buttons and combo box to the button layout
        self.button_layout.addWidget(self.load_image_button)
        self.button_layout.addWidget(self.model_combo_box)
        self.button_layout.addWidget(self.detect_button)

        # Add button layout and scroll area to the main layout
        self.main_layout.addLayout(self.button_layout)
        self.main_layout.addWidget(self.scroll_area)

        # Set the main layout to the central widget
        self.central_widget.setLayout(self.main_layout)

    # Function to handle model selection
    def handle_model_selection(self, index):
        if index == 0:  # If "Select Model" is selected
            self.detect_button.setEnabled(False)
        else:
            # Remove the "Select Model" option after selecting a valid model
            if self.model_combo_box.findText("Select Model") != -1:
                self.model_combo_box.removeItem(0)
            self.detect_button.setEnabled(True)
    
    # Function to update the info_label based on the selected model
    def update_info_label(self, model_name):
            info = self.model_info.get(model_name, "No information available for this model.")
            self.info_label.setText(info)

    # def select_model_from_combo(self, selected_model):
    #     if selected_model == "YOLO Leukemia_Cancer" :
    #         model_path = 'leukemia_yolov8.pt'
    #     elif selected_model == "YOLO 7_Cell_Class_Type" :
    #         model_path = '7_cell_class_type_yolov8.pt'
    #     elif selected_model == "YOLO Malaria" :
    #         model_path = 'malaria_yolov8.pt'
    #     elif selected_model == "YOLO WBC_Subtypes" :
    #         model_path = 'wbc_subtypes_yolov8.pt'
    #     self.model = YOLO(model_path)

    def load_images(self):
        # Open file dialog to select multiple image files
        file_names, _ = QFileDialog.getOpenFileNames(self, 'Open Image Files', '', 'Images (*.png *.jpg *.jpeg *.bmp)')
        if file_names:
            self.image_paths = file_names  # Store the paths of selected images

            # Clear previous images from both layouts
            self.clear_layout(self.image_layout)
            self.clear_layout(self.image_layout_2)
            self.clear_layout(self.text_layout)

            # Add new QLabel widgets for each loaded image
            for file_name in self.image_paths:
                label = QLabel(self)
                label.setFixedSize(700, 500)
                pixmap = QPixmap(file_name)
                pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                label.setPixmap(pixmap)
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet("border: 2px solid black;")
                self.image_layout.addWidget(label)

    def clear_layout(self, layout):
        """Helper function to clear all widgets from a layout"""
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def detect_objects(self):
        if hasattr(self, 'image_paths'):
            self.clear_layout(self.image_layout_2)
            self.clear_layout(self.text_layout)
            for image_path in self.image_paths:
                image = cv2.imread(image_path)
                results = self.model(image)
                annotated_img = results[0].plot()
                annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                # Create a new QLabel for displaying the annotated image
                label = QLabel(self)
                label.setFixedSize(700, 500)

                # Convert the image to QImage format
                height, width, channel = annotated_img.shape
                bytes_per_line = 3 * width
                q_image = QImage(annotated_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

                label.setPixmap(pixmap)
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet("border: 2px solid black;")
                self.image_layout_2.addWidget(label)

                boxes = results[0].boxes  # Accessing the bounding boxes from the first image (results[0])
                names = results[0].names  # Class names dictionary (mapping class IDs to names)
            
                detected_objects = []
            
                # Iterate over the detected boxes
                for box in boxes:
                    class_id = int(box.cls)  # Get the class ID as an integer
                    detected_objects.append(names[class_id])  # Map the class ID to the class name
            
                object_counts = Counter(detected_objects)
            
                # Display detected objects in a separate label
                detected_info = '\n'.join([f"{count} {obj}(s)" for obj, count in object_counts.items()])
                if not detected_info:
                    detected_info = "No objects detected."
                
                # Create or update the label to show the object counts
                if not hasattr(self, 'object_count_label'):
                    object_count_label = QLabel(self)
                    object_count_label.setFixedSize(400,500)
                    object_count_label.setStyleSheet("border: 2px solid black;")
                    self.text_layout.addWidget(object_count_label)  # Add it to the right-side layout
                object_count_label.setText(detected_info)
                object_count_label.setStyleSheet("border: 2px solid black; font-size: 24px;") 
                
                
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
