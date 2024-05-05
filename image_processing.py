import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QMainWindow, QAction, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QMessageBox, QInputDialog
from PyQt5.QtGui import QColor, QImage, QPixmap, QPainter, QTransform
from PyQt5.QtCore import Qt
from PIL import Image

class HomeworkTrackerWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.title_label = QLabel("Digital Image Processing")
        self.title_label.setStyleSheet("font-size: 18px; font-weight: bold;")

        self.info_label = QLabel("Course Name: Digital Image Processing \n Name Surname: Şeyda Gül Topal")
        self.info_label.setStyleSheet("margin-top: 10px; margin-bottom: 20px;")

        self.layout = QVBoxLayout(self.central_widget)
        self.layout.addWidget(self.title_label)
        self.layout.addWidget(self.info_label)

        self.create_menu()

        self.setWindowTitle("Assignment Tracking System")
        self.setGeometry(100, 100, 600, 500)

    def create_menu(self):
        menu_bar = self.menuBar()
        homework_menu = menu_bar.addMenu("Assignment")
        action = QAction("Assignment 1: Designing GUI and Implementing Basic Functionality", self)
        action.triggered.connect(lambda _, homeworknumber=1: self.show_assignment_details(homeworknumber))
        homework_menu.addAction(action)
        action = QAction("Assignment 2: Basic Image Operations and Interpolation", self)
        action.triggered.connect(lambda _, homeworknumber=2: self.show_assignment_details(homeworknumber))
        homework_menu.addAction(action)
        project_menu = menu_bar.addMenu("Midterm")
        action = QAction("Contrast Enhancement", self)
        action.triggered.connect(lambda _, tasknumber=1: self.show_midterm_details(tasknumber))
        project_menu.addAction(action)
        action = QAction("Hough Transform : Line Detection", self)
        action.triggered.connect(lambda _, tasknumber=2: self.show_midterm_details(tasknumber))
        project_menu.addAction(action)
        action = QAction("Hough Transform : Circle Detection", self)
        action.triggered.connect(lambda _, tasknumber=3: self.show_midterm_details(tasknumber))
        project_menu.addAction(action)
        action = QAction("Counting objects in the picture and extracting features", self)
        action.triggered.connect(lambda _, tasknumber=4: self.show_midterm_details(tasknumber))
        project_menu.addAction(action)

    def show_midterm_details(self, task_number):
        self.clear_layout()

        details_label = QLabel(f"Task{task_number} Details:")
        self.layout.addWidget(details_label)
        image_layout = QHBoxLayout()

        self.layout.addLayout(image_layout)

        if task_number==1:
            

            self.standart_sigmoid_button = QPushButton("Standard Sigmoid")
            self.standart_sigmoid_button.clicked.connect(lambda: self.contrast_enhancement('standard'))
            self.layout.addWidget(self.standart_sigmoid_button)

            self.steep_sigmoid_button = QPushButton("Slope Sigmoid")
            self.steep_sigmoid_button.clicked.connect(lambda: self.contrast_enhancement('slope'))
            self.layout.addWidget(self.steep_sigmoid_button)

            self.shifted_sigmoid_button = QPushButton("Shifted Sigmoid")
            self.shifted_sigmoid_button.clicked.connect(lambda: self.contrast_enhancement('shifted'))
            self.layout.addWidget(self.shifted_sigmoid_button)

        elif task_number==2:
            
            self.line_detection_button = QPushButton("Line Detection")
            img = cv2.imread ('road.jpg')
            self.line_detection_button.clicked.connect(lambda: self.line_detection(img))
            self.layout.addWidget(self.line_detection_button)

        elif task_number==3:
            image = cv2.imread('eye.jpg')
            self.circle_detection_button = QPushButton("Circle Detection")
            self.circle_detection_button.clicked.connect(lambda: self.circle_detection(image))
            self.layout.addWidget(self.circle_detection_button)
        elif task_number==4:
            image = cv2.imread('image.jpeg')
            self.object_detection_button = QPushButton("Counting objects in the picture and extracting features")
            self.object_detection_button.clicked.connect(lambda: self.object_detection(image))
            self.layout.addWidget(self.object_detection_button)
        
            
        self.current_image_size_label = QLabel()
        self.layout.addWidget(self.current_image_size_label)

    @staticmethod
    def standard_sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def shifted_sigmoid(x , c):
        return 1 / (1 + np.exp(-(x - c)))

    @staticmethod
    def slope_sigmoid(x, a):
        return 1 / (1 + np.exp(-a * x))

    def contrast_enhancement(self,sigmoid_func):
        image = np.array(Image.open("contrast.png"))

        if len(image.shape) == 3:
            image = np.mean(image, axis=2)
        normalized_image = image / 255.0
        if sigmoid_func == 'standard':
                enhanced_image = self.standard_sigmoid(normalized_image)
                
        elif sigmoid_func=='slope':
                a, _ = QInputDialog.getInt(self, 'Slope', 'Input slope value:', 1, 5, 10)
                enhanced_image = self.slope_sigmoid(normalized_image, a)

        elif sigmoid_func== 'shifted':
                c, _ = QInputDialog.getInt(self, 'Shift', 'Input shift value:', 1, 5, 10)
                enhanced_image = self.shifted_sigmoid(normalized_image, c)

        enhanced_image = (enhanced_image * 255).astype(np.uint8)
        self.display_images(image,enhanced_image)        

        

    def line_detection (self,img):
        gray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
        cv2. imshow ('edges', edges)
        lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
        for line in lines:
            x1,y1,x2, y2 = line [0]
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.imshow ('image', img)
        k = cv2.waitKey (0)

    def circle_detection(self,image): #eye pupil detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray_blurred = cv2.GaussianBlur(gray, (15, 15), 0)

        circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                param1=50, param2=30, minRadius=10, maxRadius=50)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

            for (x, y, r) in circles:
                cv2.circle(image, (x, y), r, (0, 255, 0), 4)

            cv2.imshow('Detected Pupils', image)
            cv2.waitKey(0)
        else:
            print("No circles detected.")

    def object_detection (self,image):
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([90, 255, 255])

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_green, upper_green)

        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        properties = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                length = h
                width = w
                diagonal = np.sqrt(w**2 + h**2)
                roi = image[y:y+h, x:x+w]
                energy = np.sum(roi.astype("float") ** 2)
                entropy = -np.sum(np.multiply(roi.astype("float"), np.log2(roi.astype("float") + np.finfo(float).eps)))
                mean = np.mean(roi)
                median = np.median(roi)
                
                properties.append({
                    "Center": (center_x, center_y),
                    "Length": length,
                    "Width": width,
                    "Diagonal": diagonal,
                    "Energy": energy,
                    "Entropy": entropy,
                    "Mean": mean,
                    "Median": median
                })

        df = pd.DataFrame(properties)

        df.to_excel('object_features.xlsx', index=False)

        QMessageBox.information(self, "Information", "Features successfully saved to Excel file.")

    def upload_midterm_image(self):
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpeg *.jpg *.bmp)")

            if file_path:
                self.original_image = np.array(Image.open(file_path))
                
            else:
                QMessageBox.warning(self, "Warning", "Select Image.")

    def display_images(self, original_image, enhanced_image):  
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title('Orjinal Görüntü')
        plt.subplot(1, 2, 2)
        plt.imshow(enhanced_image, cmap='gray')
        plt.title('Contrast Enhanced Image')
        plt.show()

    def display_original_image(self, original_image):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title('Orjinal Görüntü')

    def clear_layout(self):
        for i in reversed(range(self.layout.count())):
            item = self.layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()

    def upload_assignment_image(self, homework_number):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpeg *.jpg *.bmp)")

        if file_path:
            image = QImage(file_path)
            self.original_image = QPixmap.fromImage(image)
            self.display_image(self.original_image)
            self.current_image_size_label.setText(f"Current Size: {self.original_image.width()} x {self.original_image.height()}")
        else:
            QMessageBox.warning(self, "Warning", "Select Image.")

    def display_image(self, image):
        self.image_label.setPixmap(image.scaled(512, 512, Qt.KeepAspectRatio))

    def zoom_in(self):
        if hasattr(self, 'original_image'):
            width = int(self.original_image.width() * 0.5)
            height = int(self.original_image.height() * 0.5)
            self.original_image = self.original_image.scaled(width, height, Qt.KeepAspectRatio)
            self.display_image(self.original_image)

    def zoom_out(self):
        if hasattr(self, 'original_image'):
            width = int(self.original_image.width() * 0.50)
            height = int(self.original_image.height() * 0.50)
            self.original_image = self.original_image.scaled(width, height, Qt.KeepAspectRatio)
            self.display_image(self.original_image)

    def rotate_image(self):
        if hasattr(self, 'original_image'):
            degree, ok = QInputDialog.getInt(self, 'Rotate', 'Input Degree:', 0, -360, 360)
            if ok:
                transform = QTransform().rotate(degree)
                self.original_image = self.original_image.transformed(transform)
                self.display_image(self.original_image)

    def change_image_size(self):
        if hasattr(self, 'original_image'):
            new_size, ok = QInputDialog.getText(self, 'Change Image Size', 'New Size (width x height):')
            if ok:
                new_size = new_size.split('x')
                if len(new_size) == 1:
                    new_width = int(new_size[0])
                    new_height = int(new_size[0])
                elif len(new_size) == 2:
                    new_width = int(new_size[0])
                    new_height = int(new_size[1])
                else:
                    QMessageBox.warning(self, "Warning", "Invalid size information.")
                    return
                if new_width > 0 and new_height > 0:
                    self.original_image = self.resize_image(self.original_image, new_width, new_height)
                    self.display_image(self.original_image)
                    self.current_image_size_label.setText(f"Currebt Size: {self.original_image.width()} x {self.original_image.height()}\nNew Size: {new_width} x {new_height}")
                else:
                    QMessageBox.warning(self, "Warning", "The width and height values ​​must be greater than zero.")
            else:
                QMessageBox.warning(self, "Warning", "Valid size information was not entered.")

    def resize_image(self, image, width, height):
        result_image = QPixmap(width, height)
        result_image.fill(Qt.black)
        painter = QPainter(result_image)
        target_x = (result_image.width() - image.width()) // 2
        target_y = (result_image.height() - image.height()) // 2
        painter.drawPixmap(target_x, target_y, image)
        painter.end()
        return result_image

    def bilinear_interpolation(self, image, new_width, new_height):
        old_height, old_width = image.height(), image.width()
        image = image.toImage()
        x_ratio = old_width / new_width
        y_ratio = old_height / new_height

        new_image = QImage(new_width, new_height, QImage.Format_ARGB32)

        for y in range(new_height):
            for x in range(new_width):
                x_old = (x + 0.5) * x_ratio - 0.5
                y_old = (y + 0.5) * y_ratio - 0.5

                x0 = int(x_old)
                y0 = int(y_old)
                x1 = min(x0 + 1, old_width - 1)
                y1 = min(y0 + 1, old_height - 1)

                dx = x_old - x0
                dy = y_old - y0

                c00 = image.pixelColor(x0, y0)
                c10 = image.pixelColor(x1, y0)
                c01 = image.pixelColor(x0, y1)
                c11 = image.pixelColor(x1, y1)

                new_color = c00.toRgb().toTuple()
                for i in range(3): 
                    new_color_channel = (1 - dx) * (1 - dy) * new_color[i] + \
                                        dx * (1 - dy) * c10.toRgb().toTuple()[i] + \
                                        (1 - dx) * dy * c01.toRgb().toTuple()[i] + \
                                        dx * dy * c11.toRgb().toTuple()[i]
                    new_color_channel = max(0, min(int(new_color_channel), 255))  
                    new_color[i] = new_color_channel

                new_image.setPixelColor(x, y, QColor(*new_color))

        return QPixmap.fromImage(new_image)

    def upload_resized_image(self, homework_number, new_width, new_height):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpeg *.jpg *.bmp)")

        if file_path:
            image = QPixmap(file_path)
            resized_image = self.bilinear_interpolation(image, new_width, new_height)
            self.display_image(resized_image)
            self.current_image_size_label.setText(f"Current Size: {resized_image.width()} x {resized_image.height()}")
        else:
            QMessageBox.warning(self, "Warning", "The image was not selected.")


    def show_assignment_details(self, homework_number):
        self.clear_layout()

        details_label = QLabel(f"Assignment{homework_number} Details:")
        self.layout.addWidget(details_label)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(512, 512)
        self.layout.addWidget(self.image_label)

        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(lambda: self.upload_assignment_image(homework_number))
        self.layout.addWidget(self.upload_button)

        self.zoom_in_button = QPushButton("Zoom In")
        self.zoom_in_button.clicked.connect(lambda: self.zoom_in())
        self.layout.addWidget(self.zoom_in_button)

        self.zoom_out_button = QPushButton("Zoom Out")
        self.zoom_out_button.clicked.connect(lambda: self.zoom_out())
        self.layout.addWidget(self.zoom_out_button)

        self.rotate_button = QPushButton("Turn")
        self.rotate_button.clicked.connect(lambda: self.rotate_image())
        self.layout.addWidget(self.rotate_button)

        self.resize_button = QPushButton("Change Image Size")
        self.resize_button.clicked.connect(lambda: self.change_image_size())
        self.layout.addWidget(self.resize_button)

        self.current_image_size_label = QLabel()
        self.layout.addWidget(self.current_image_size_label)

       
if __name__ == "__main__":
    app = QApplication(sys.argv)
    homework_tracker = HomeworkTrackerWindow()
    homework_tracker.show()
    sys.exit(app.exec_())