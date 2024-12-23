import sys
import os
import random
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QPushButton, QLineEdit
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PIL import Image, ImageOps
import tensorflow as tf
import pyqtgraph as pg


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Automatically load the trained model
model = tf.keras.models.load_model(resource_path("cnn_dye_loading_model.h5"))


# Function to calculate loading based on the Langmuir isotherm
def langmuir_loading(transparency, K=3.0, Q_max=1.0):
    transparency_scaled = 100 - transparency
    C = transparency_scaled / 100 * 10  # Rescale transparency to concentration range
    loading = (Q_max * C) / (K + C)
    return loading * 100  # Convert to percentage

# Function to simulate bioplastic transparency over time
def calculate_transparency(time_hours):
    time_hours += time_hours * random.uniform(-0.01, 0.01)
    transparency = 100 * np.exp(-0.2 * time_hours)
    return transparency

# Function to calculate dye loading using the ML model
def predict_loading(image):
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    predicted_loading = model.predict(image_array)[0][0]
    return predicted_loading

# Function to apply random transformations to the image
def apply_transformations(image):
    angle = random.uniform(-30, 30)
    transformed_image = image.rotate(angle)

    if random.choice([True, False]):
        transformed_image = ImageOps.mirror(transformed_image)
    if random.choice([True, False]):
        transformed_image = ImageOps.flip(transformed_image)

    return transformed_image

# Function to create hybrid images based on transparency
def create_hybrid_image(transparency):
    alpha = transparency / 100
    unloaded_image = Image.open(resource_path("biofilm2.jpg")).resize((200, 200))
    blue_image = Image.open(resource_path("biofilm1.jpg")).resize((200, 200))
    hybrid_image = Image.blend(unloaded_image, blue_image, alpha)
    return hybrid_image

# Main window class using PyQt5
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Bioplastic Loading Simulation")
        self.setGeometry(100, 100, 800, 600)

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        layout = QVBoxLayout(self.main_widget)

        # Time input
        self.time_label = QLabel("Time in Dye Solution (hours):", self)
        self.time_input = QLineEdit(self)
        layout.addWidget(self.time_label)
        layout.addWidget(self.time_input)

        # Simulate button
        self.simulate_button = QPushButton("Simulate Loading", self)
        self.simulate_button.clicked.connect(self.simulate_bioplastic_loading)
        layout.addWidget(self.simulate_button)

        # Image display label
        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

        # PyQtGraph plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Loading (%)')
        self.plot_widget.setLabel('bottom', 'Time (hours)')
        self.plot_widget.setXRange(-0.5, 15.5)
        self.plot_widget.setYRange(0, 100)
        layout.addWidget(self.plot_widget)

        self.real_plot_data = self.plot_widget.plot([], [], pen=None, symbol='o', symbolBrush='grey', name='Real Loading')
        self.plot_data = self.plot_widget.plot([], [], pen=None, symbol='o', symbolBrush='b', name='Predicted Loading')


        self.time_points = []
        self.real_loadings = []
        self.predicted_loadings = []

    def simulate_bioplastic_loading(self):
        try:
            time_hours = float(self.time_input.text())
        except ValueError:
            print("Invalid input: Please enter a valid number of hours.")
            return

        transparency = calculate_transparency(time_hours)
        hybrid_image = create_hybrid_image(transparency)
        transformed_image = apply_transformations(hybrid_image)
        predicted_loading = predict_loading(transformed_image)
        real_loading = langmuir_loading(transparency)

        # Convert PIL image to QImage manually
        transformed_image = transformed_image.convert("RGBA")
        data = transformed_image.tobytes("raw", "RGBA")
        qimage = QImage(data, transformed_image.size[0], transformed_image.size[1], QImage.Format_RGBA8888)

        # Display image
        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio))

        # Update the plot with real and predicted loading
        self.time_points.append(time_hours)
        self.real_loadings.append(real_loading)
        self.predicted_loadings.append(predicted_loading)

        self.plot_data.setData(self.time_points, self.predicted_loadings)
        self.real_plot_data.setData(self.time_points, self.real_loadings)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
