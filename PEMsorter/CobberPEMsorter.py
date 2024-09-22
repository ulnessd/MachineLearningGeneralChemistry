import sys
import os
import cv2
import numpy as np
import pickle
import random
import threading
import time

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton,
                             QFileDialog, QVBoxLayout, QHBoxLayout, QMessageBox,
                             QProgressBar, QTextEdit)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal, QObject

import tensorflow as tf
from tensorflow.keras.models import load_model

from sklearn.preprocessing import LabelEncoder

# ---------------------------
# Image Generation Functions
# ---------------------------

def generate_bead_image(image_size=(256, 256),
                        small_bead_radius=(8, 10),
                        large_bead_radius=(18, 20),
                        num_beads=50,
                        category='mixed',
                        small_prob=0.5):
    """
    Generates a synthetic Polymer Electron Microscopy (PEM) image with beads.

    Parameters:
        image_size (tuple): Size of the image (height, width).
        small_bead_radius (tuple): Min and max radius for small beads.
        large_bead_radius (tuple): Min and max radius for large beads.
        num_beads (int): Number of beads to draw.
        category (str): Category of beads - 'pure_small', 'pure_large', or 'mixed'.
        small_prob (float): Probability of a bead being small in mixed images.

    Returns:
        np.ndarray: Generated synthetic PEM image.
    """
    # Validate category input
    if category not in ['pure_small', 'pure_large', 'mixed']:
        raise ValueError("Invalid category. Choose from 'pure_small', 'pure_large', or 'mixed'.")

    # Create a blank grayscale image (black background)
    image = np.zeros(image_size, dtype=np.uint8)  # Black background

    existing_beads = []  # To keep track of bead positions and radii for overlap checking

    for _ in range(num_beads):
        # Decide bead size based on category
        if category == 'pure_small':
            radius = random.randint(small_bead_radius[0], small_bead_radius[1])
        elif category == 'pure_large':
            radius = random.randint(large_bead_radius[0], large_bead_radius[1])
        elif category == 'mixed':
            if random.random() < small_prob:
                radius = random.randint(small_bead_radius[0], small_bead_radius[1])
            else:
                radius = random.randint(large_bead_radius[0], large_bead_radius[1])

        # Random position ensuring the bead is fully within the image boundaries
        x = random.randint(radius, image_size[1] - radius)
        y = random.randint(radius, image_size[0] - radius)

        # Optional: Prevent excessive overlapping (simple collision detection)
        overlap = False
        for bead in existing_beads:
            distance = np.sqrt((x - bead['x'])**2 + (y - bead['y'])**2)
            if distance < (radius + bead['radius'] + 2):  # Adding a small buffer
                overlap = True
                break
        if overlap:
            continue  # Skip this bead to prevent overlap

        # Add bead to existing_beads list
        existing_beads.append({'x': x, 'y': y, 'radius': radius})

        # Create a bead mask with a base intensity higher than the background
        bead = np.zeros((2 * radius, 2 * radius), dtype=np.uint8)  # Start with black
        base_intensity = 50  # Increased base intensity for better visibility
        cv2.circle(bead, (radius, radius), radius, base_intensity, -1)  # Filled circle

        # Apply radial gradient for shading to simulate depth
        for i in range(1, radius):
            intensity = base_intensity + int((i / radius) * 200)  # Gradient from base_intensity to base_intensity + 200
            intensity = min(intensity, 255)  # Ensure intensity does not exceed 255
            cv2.circle(bead, (radius, radius), radius - i, intensity, 1)  # Draw concentric circles

        # Define the region of interest (ROI) on the main image where the bead will be placed
        roi = image[y - radius:y + radius, x - radius:x + radius]

        # Ensure ROI dimensions match bead dimensions
        if roi.shape[0] != 2 * radius or roi.shape[1] != 2 * radius:
            # Skip drawing beads that would go out of bounds
            continue

        # Combine the bead with the ROI using cv2.max to brighten the bead area
        combined = cv2.max(roi, bead)
        image[y - radius:y + radius, x - radius:x + radius] = combined

    # Add Gaussian noise to simulate real EM image imperfections
    noise_sigma = 0.15  # Adjusted standard deviation of the noise for better visibility
    noise = np.random.normal(0, noise_sigma, image_size).astype(np.uint8)
    noisy_image = cv2.add(image, noise)

    return noisy_image

# ---------------------------
# WorkerSignals Class
# ---------------------------

class WorkerSignals(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    log = pyqtSignal(str)
    display_image = pyqtSignal(np.ndarray)  # Emit image data for display
    update_prediction = pyqtSignal(str)  # Emit prediction result
    actual_predicted = pyqtSignal(str, str)  # Emit actual and predicted classes

# ---------------------------
# SorterThread Class
# ---------------------------

class SorterThread(threading.Thread):
    def __init__(self, model, le, signals, num_iterations=200, delay=0.1):
        threading.Thread.__init__(self)
        self.model = model
        self.le = le
        self.signals = signals
        self.num_iterations = num_iterations
        self.delay = delay  # Delay between iterations in seconds

        # Define directories for saving images
        self.output_dirs = {
            'pure_small': 'pure_small',
            'pure_large': 'pure_large',
            'acceptable_small': 'acceptable_small',
            'acceptable_large': 'acceptable_large',
            'reject': 'reject'
        }
        # Create directories if they don't exist
        for dir_name in self.output_dirs.values():
            os.makedirs(dir_name, exist_ok=True)

    def run(self):
        try:
            total_errors = 0
            for i in range(1, self.num_iterations + 1):
                # Determine category based on probabilities
                rand = random.random()
                if rand < 0.25:
                    category = 'pure_small'
                    small_prob = 0.0  # Not used
                elif rand < 0.50:
                    category = 'pure_large'
                    small_prob = 0.0  # Not used
                else:
                    category = 'mixed'
                    small_prob = random.uniform(0.05, 0.95)

                # Determine sorting_bin based on small_prob
                if category == 'mixed':
                    if small_prob >= 0.92:
                        sorting_bin = 'acceptable_small'
                    elif small_prob <= 0.08:
                        sorting_bin = 'acceptable_large'
                    else:
                        sorting_bin = 'reject'
                else:
                    sorting_bin = category  # 'pure_small' or 'pure_large'

                # Generate image
                img = generate_bead_image(
                    image_size=(256, 256),
                    small_bead_radius=(8, 10),
                    large_bead_radius=(18, 20),
                    num_beads=random.randint(40, 90),
                    category=category,
                    small_prob=small_prob
                )

                # Predict class
                prediction = self.predict_class(img)

                # Emit image for display
                self.signals.display_image.emit(img)

                # Emit actual and predicted classes for logging
                self.signals.actual_predicted.emit(sorting_bin, prediction)

                # Check for mismatch based on user's criteria
                if self.is_bad_error(sorting_bin, prediction):
                    total_errors += 1

                # Save image to appropriate folder
                save_dir = self.output_dirs.get(prediction, 'reject')  # Default to 'reject' if not found
                filename = f"batch{str(i).zfill(4)}.png"
                filepath = os.path.join(save_dir, filename)
                cv2.imwrite(filepath, img)

                # Emit progress
                progress = int((i / self.num_iterations) * 100)
                self.signals.progress.emit(progress)

                # Delay
                time.sleep(self.delay)

            # Calculate percent identification error
            percent_error = (total_errors / self.num_iterations) * 100
            self.signals.log.emit(f"Percent Identification Error: {percent_error:.2f}%")

            self.signals.finished.emit()
        except Exception as e:
            self.signals.error.emit(str(e))

    def predict_class(self, img):
        """
        Predicts the class of the given image using the loaded model.

        Parameters:
            img (np.ndarray): Grayscale image.

        Returns:
            str: Predicted class label.
        """
        # Preprocess image
        img_resized = cv2.resize(img, (256, 256))
        img_normalized = img_resized / 255.0
        img_expanded = np.expand_dims(img_normalized, axis=-1)  # Add channel dimension
        img_input = np.expand_dims(img_expanded, axis=0)  # Add batch dimension

        # Predict
        preds = self.model.predict(img_input)
        pred_class_idx = np.argmax(preds, axis=1)[0]
        pred_class = self.le.inverse_transform([pred_class_idx])[0]
        return pred_class

    def is_bad_error(self, actual, predicted):
        """
        Determines if the prediction is a bad error based on user's criteria.

        Parameters:
            actual (str): Actual class label.
            predicted (str): Predicted class label.

        Returns:
            bool: True if it's a bad error, False otherwise.
        """
        # Correct prediction
        if actual == predicted:
            return False

        # Bad Errors:
        # 1. Predict "reject" when actual is not "reject".
        if predicted == 'reject' and actual != 'reject':
            return True

        # 2. Predict not "reject" when actual is "reject".
        if actual == 'reject' and predicted != 'reject':
            return True

        # 3. Predict "pure_small" when actual is "pure_large" and vice versa.
        if (actual == 'pure_small' and predicted == 'pure_large') or \
           (actual == 'pure_large' and predicted == 'pure_small'):
            return True

        # All other misclassifications are considered acceptable
        return False

# ---------------------------
# PEMClassifierGUI Class
# ---------------------------

class PEMClassifierGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cobber PEM Inspector and Sorter")
        self.setGeometry(100, 100, 1000, 600)

        # Initialize model and label encoder
        self.model = None
        self.le = None

        # Initialize UI components
        self.initUI()

    def initUI(self):
        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layouts
        main_layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        display_layout = QHBoxLayout()
        log_layout = QVBoxLayout()

        # Buttons
        self.load_model_button = QPushButton("Load Model")
        self.load_model_button.clicked.connect(self.load_trained_model)

        self.inspect_batch_button = QPushButton("Inspect Batch")
        self.inspect_batch_button.clicked.connect(self.inspect_batch)
        self.inspect_batch_button.setEnabled(False)

        self.run_sort_button = QPushButton("Run Sort")
        self.run_sort_button.clicked.connect(self.run_sort)
        self.run_sort_button.setEnabled(False)

        button_layout.addWidget(self.load_model_button)
        button_layout.addWidget(self.inspect_batch_button)
        button_layout.addWidget(self.run_sort_button)

        # Image Display
        self.image_label = QLabel("Generated Image Preview")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(500, 500)
        self.image_label.setStyleSheet("border: 1px solid black;")

        # Log Display
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)

        display_layout.addWidget(self.image_label)
        display_layout.addWidget(self.log_text)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

        # Add layouts to main layout
        main_layout.addLayout(button_layout)
        main_layout.addLayout(display_layout)
        main_layout.addWidget(self.progress_bar)

        self.central_widget.setLayout(main_layout)

    def load_trained_model(self):
        options = QFileDialog.Options()
        model_file, _ = QFileDialog.getOpenFileName(self, "Select Trained Model", "",
                                                    "Keras Model Files (*.keras);;All Files (*)", options=options)
        if model_file:
            try:
                self.model = load_model(model_file)
                # Load label encoder
                le_file, _ = QFileDialog.getOpenFileName(self, "Select Label Encoder", "",
                                                        "Pickle Files (*.pkl);;All Files (*)", options=options)
                if le_file:
                    with open(le_file, 'rb') as f:
                        self.le = pickle.load(f)
                else:
                    QMessageBox.warning(self, "Label Encoder Missing", "Label encoder file not selected.")
                    return

                self.log_text.append(f"Model loaded from: {model_file}")
                self.log_text.append(f"Label encoder loaded from: {le_file}")
                self.inspect_batch_button.setEnabled(True)
                self.run_sort_button.setEnabled(True)
                QMessageBox.information(self, "Model Loaded", "Model and label encoder loaded successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Loading Error", f"Failed to load model or label encoder:\n{str(e)}")

    def inspect_batch(self):
        if not self.model or not self.le:
            QMessageBox.warning(self, "Model Not Loaded", "Please load a trained model before inspection.")
            return

        # Determine category based on probabilities
        rand = random.random()
        if rand < 0.25:
            category = 'pure_small'
            small_prob = 0.0  # Not used
        elif rand < 0.50:
            category = 'pure_large'
            small_prob = 0.0  # Not used
        else:
            category = 'mixed'
            small_prob = random.uniform(0.05, 0.95)

        # Determine sorting_bin based on small_prob
        if category == 'mixed':
            if small_prob >= 0.92:
                sorting_bin = 'acceptable_small'
            elif small_prob <= 0.08:
                sorting_bin = 'acceptable_large'
            else:
                sorting_bin = 'reject'
        else:
            sorting_bin = category  # 'pure_small' or 'pure_large'

        # Generate image
        img = generate_bead_image(
            image_size=(256, 256),
            small_bead_radius=(8, 10),
            large_bead_radius=(18, 20),
            num_beads=random.randint(40, 90),
            category=category,
            small_prob=small_prob
        )

        # Predict class
        prediction = self.predict_class(img)

        # Display image
        self.display_image(img)

        # Log prediction with color coding
        if sorting_bin == prediction:
            # Match - Green
            self.log_text.append(f"<span style='color: green;'>Inspect Batch: Actual - {sorting_bin}, Predicted - {prediction}</span>")
        elif self.is_bad_error(sorting_bin, prediction):
            # Bad Error - Red
            self.log_text.append(f"<span style='color: red;'>Inspect Batch: Actual - {sorting_bin}, Predicted - {prediction}</span>")
        else:
            # Acceptable Error - Blue
            self.log_text.append(f"<span style='color: blue;'>Inspect Batch: Actual - {sorting_bin}, Predicted - {prediction}</span>")

    def run_sort(self):
        if not self.model or not self.le:
            QMessageBox.warning(self, "Model Not Loaded", "Please load a trained model before running sort.")
            return

        # Disable buttons during sorting
        self.run_sort_button.setEnabled(False)
        self.inspect_batch_button.setEnabled(False)
        self.load_model_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Initialize signals
        self.signals = WorkerSignals()
        self.signals.progress.connect(self.update_progress)
        self.signals.finished.connect(self.sorting_finished)
        self.signals.error.connect(self.sorting_error)
        self.signals.log.connect(self.log_text.append)
        self.signals.display_image.connect(self.display_image)
        self.signals.actual_predicted.connect(self.log_actual_predicted)

        # Initialize counters for error calculation
        self.total_predictions = 0
        self.total_bad_errors = 0

        # Start sorter thread
        self.sorter_thread = SorterThread(self.model, self.le, self.signals, num_iterations=200, delay=0.1)
        self.sorter_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def sorting_finished(self):
        # Calculate percent identification error
        if self.total_predictions > 0:
            percent_error = (self.total_bad_errors / self.total_predictions) * 100
            self.log_text.append(f"<b>Percent Identification Error: {percent_error:.2f}%</b>")
        else:
            self.log_text.append("<b>No predictions were made.</b>")

        # Enable buttons
        self.run_sort_button.setEnabled(True)
        self.inspect_batch_button.setEnabled(True)
        self.load_model_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.information(self, "Sorting Complete", "Batch sorting of 200 images completed.")

    def sorting_error(self, error_msg):
        # Enable buttons
        self.run_sort_button.setEnabled(True)
        self.inspect_batch_button.setEnabled(True)
        self.load_model_button.setEnabled(True)
        self.progress_bar.setVisible(False)

        # Log error
        self.log_text.append(f"<span style='color: red;'>Error during sorting: {error_msg}</span>")
        QMessageBox.critical(self, "Sorting Error", f"An error occurred during sorting:\n{error_msg}")

    def log_actual_predicted(self, actual, predicted):
        """
        Logs the actual and predicted classes with color coding.
        Also tracks bad errors for statistical reporting.
        """
        self.total_predictions += 1
        if actual == predicted:
            # Correct Prediction - Green
            self.log_text.append(f"<span style='color: green;'>Image {self.total_predictions}: Actual - {actual}, Predicted - {predicted}</span>")
        elif self.is_bad_error(actual, predicted):
            # Bad Prediction - Red
            self.log_text.append(f"<span style='color: red;'>Image {self.total_predictions}: Actual - {actual}, Predicted - {predicted}</span>")
            self.total_bad_errors += 1
        else:
            # Acceptable Prediction - Blue
            self.log_text.append(f"<span style='color: blue;'>Image {self.total_predictions}: Actual - {actual}, Predicted - {predicted}</span>")

    def is_bad_error(self, actual, predicted):
        """
        Determines if the prediction is a bad error based on user's criteria.

        Parameters:
            actual (str): Actual class label.
            predicted (str): Predicted class label.

        Returns:
            bool: True if it's a bad error, False otherwise.
        """
        # Bad Errors:
        # 1. Predict "reject" when actual is not "reject".
        if predicted == 'reject' and actual != 'reject':
            return True

        # 2. Predict not "reject" when actual is "reject".
        if actual == 'reject' and predicted != 'reject':
            return True

        # 3. Predict "pure_small" when actual is "pure_large" and vice versa.
        if (actual == 'pure_small' and predicted == 'pure_large') or \
           (actual == 'pure_large' and predicted == 'pure_small'):
            return True

        # All other misclassifications are considered acceptable
        return False

    def predict_class(self, img):
        """
        Predicts the class of the given image using the loaded model.

        Parameters:
            img (np.ndarray): Grayscale image.

        Returns:
            str: Predicted class label.
        """
        # Preprocess image
        img_resized = cv2.resize(img, (256, 256))
        img_normalized = img_resized / 255.0
        img_expanded = np.expand_dims(img_normalized, axis=-1)  # Add channel dimension
        img_input = np.expand_dims(img_expanded, axis=0)  # Add batch dimension

        # Predict
        preds = self.model.predict(img_input)
        pred_class_idx = np.argmax(preds, axis=1)[0]
        pred_class = self.le.inverse_transform([pred_class_idx])[0]
        return pred_class

    def display_image(self, img):
        """
        Displays the given image in the GUI.

        Parameters:
            img (np.ndarray): Grayscale image.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        height, width, channel = img_rgb.shape
        bytes_per_line = 3 * width
        q_img = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

# ---------------------------
# Main Function
# ---------------------------

def main():
    app = QApplication(sys.argv)
    gui = PEMClassifierGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
