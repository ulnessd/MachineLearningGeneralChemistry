import sys
import os
import cv2
import numpy as np
import pickle
import threading

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton,
                             QFileDialog, QVBoxLayout, QHBoxLayout, QListWidget,
                             QListWidgetItem, QMessageBox, QProgressBar, QGridLayout,
                             QTabWidget, QTextEdit, QLineEdit)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal, QObject

import pyqtgraph as pg

import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split  # <-- Added Import

import pandas as pd
from tqdm import tqdm


class WorkerSignals(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    log = pyqtSignal(str)


class TrainerThread(threading.Thread):
    def __init__(self, image_dir, metadata_file, signals):
        threading.Thread.__init__(self)
        self.image_dir = image_dir
        self.metadata_file = metadata_file
        self.signals = signals

    def run(self):
        try:
            self.signals.log.emit("Loading data...")
            X, y = self.load_data(self.image_dir, self.metadata_file, img_size=(256, 256))
            if X.size == 0:
                self.signals.error.emit("No images loaded. Please check the directory and metadata.")
                return

            self.signals.log.emit("Preprocessing labels...")
            y_encoded, le = self.preprocess_labels(y)

            self.signals.log.emit("Splitting data into training, validation, and testing sets...")
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

            self.signals.log.emit("Calculating class weights...")
            y_integers = np.argmax(y_train, axis=1)
            class_weights_vals = class_weight.compute_class_weight(
                'balanced',
                classes=np.unique(y_integers),
                y=y_integers
            )
            class_weights = {i: class_weights_vals[i] for i in range(len(class_weights_vals))}

            self.signals.log.emit("Building the CNN model...")
            input_shape = X_train.shape[1:]  # (256, 256, 1)
            num_classes = y_encoded.shape[1]  # 5
            model = self.build_cnn_model(input_shape, num_classes)

            self.signals.log.emit("Starting model training...")
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ModelCheckpoint('best_cnn_model.keras', save_best_only=True)  # Updated Extension
            ]

            history = model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_val, y_val),
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=0
            )

            self.signals.log.emit("Training complete. Evaluating on test data...")
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            self.signals.log.emit(f"Test Accuracy: {test_acc * 100:.2f}%")

            # Save the label encoder
            with open('label_encoder.pkl', 'wb') as f:
                pickle.dump(le, f)
            self.signals.log.emit("Label encoder saved as 'label_encoder.pkl'.")

            self.signals.finished.emit()
        except Exception as e:
            self.signals.error.emit(str(e))

    def load_data(self, image_dir, metadata_file, img_size=(256, 256)):
        metadata = pd.read_csv(metadata_file)
        X = []
        y = []

        for index, row in tqdm(metadata.iterrows(), total=metadata.shape[0]):
            img_path = os.path.join(image_dir, row['filename'])
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            img = img / 255.0
            img = np.expand_dims(img, axis=-1)
            X.append(img)
            y.append(row['sorting_bin'])

        X = np.array(X)
        y = np.array(y)
        return X, y

    def preprocess_labels(self, y):
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        y_encoded = tf.keras.utils.to_categorical(y_encoded)
        return y_encoded, le

    def build_cnn_model(self, input_shape, num_classes):
        model = Sequential()

        # First Convolutional Block
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))

        # Second Convolutional Block
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))

        # Third Convolutional Block
        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))

        # Fully Connected Layers
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(512, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

        # Compile the model with updated optimizer argument
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Updated Parameter
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model


class ClassifierThread(threading.Thread):
    def __init__(self, image_paths, model, le, signals):
        threading.Thread.__init__(self)
        self.image_paths = image_paths
        self.model = model
        self.le = le
        self.signals = signals

    def run(self):
        try:
            predictions = []
            probabilities = []
            for idx, img_path in enumerate(self.image_paths):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    self.signals.log.emit(f"Failed to load image: {img_path}")
                    continue
                img = cv2.resize(img, (256, 256))
                img = img / 255.0
                img = np.expand_dims(img, axis=-1)  # Add channel dimension
                img = np.expand_dims(img, axis=0)  # Add batch dimension

                # Predict
                preds = self.model.predict(img)
                pred_class = self.le.inverse_transform([np.argmax(preds)])
                pred_prob = preds[0]

                predictions.append(pred_class[0])
                probabilities.append(pred_prob)

                # Update progress
                progress = int(((idx + 1) / len(self.image_paths)) * 100)
                self.signals.progress.emit(progress)
                self.signals.log.emit(
                    f"Classified {os.path.basename(img_path)}: {pred_class[0]} ({np.max(pred_prob) * 100:.2f}%)")

            self.signals.finished.emit((predictions, probabilities))
        except Exception as e:
            self.signals.error.emit(str(e))


class PEMClassifierGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PEM Image Classifier and Trainer")
        self.setGeometry(100, 100, 1400, 800)

        # Initialize model and label encoder
        self.model = None
        self.le = None

        # Initialize UI components
        self.initUI()

    def initUI(self):
        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Main layout with tabs
        self.tabs = QTabWidget()
        self.training_tab = QWidget()
        self.classification_tab = QWidget()

        self.tabs.addTab(self.training_tab, "Training")
        self.tabs.addTab(self.classification_tab, "Classification")

        # Setup each tab
        self.init_training_tab()
        self.init_classification_tab()

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)
        self.central_widget.setLayout(main_layout)

    def init_training_tab(self):
        layout = QVBoxLayout()

        # Data selection
        data_layout = QHBoxLayout()
        self.data_path_edit = QLineEdit()
        self.data_path_edit.setReadOnly(True)
        browse_button = QPushButton("Select Data Folder")
        browse_button.clicked.connect(self.browse_data_folder)
        data_layout.addWidget(QLabel("Data Folder:"))
        data_layout.addWidget(self.data_path_edit)
        data_layout.addWidget(browse_button)

        # Train button
        self.train_button = QPushButton("Start Training")
        self.train_button.clicked.connect(self.start_training)
        self.train_button.setEnabled(False)

        # Progress bar
        self.training_progress = QProgressBar()
        self.training_progress.setValue(0)
        self.training_progress.setVisible(False)

        # Log area
        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)

        # Training metrics plots
        self.loss_plot = pg.PlotWidget(title="Loss Over Epochs")
        self.loss_plot.setLabel('left', 'Loss')
        self.loss_plot.setLabel('bottom', 'Epoch')
        self.loss_plot.showGrid(x=True, y=True)
        self.loss_curve = self.loss_plot.plot(pen='r', name='Training Loss')
        self.val_loss_curve = self.loss_plot.plot(pen='b', name='Validation Loss')

        self.acc_plot = pg.PlotWidget(title="Accuracy Over Epochs")
        self.acc_plot.setLabel('left', 'Accuracy')
        self.acc_plot.setLabel('bottom', 'Epoch')
        self.acc_plot.showGrid(x=True, y=True)
        self.acc_curve = self.acc_plot.plot(pen='r', name='Training Accuracy')
        self.val_acc_curve = self.acc_plot.plot(pen='b', name='Validation Accuracy')

        # Arrange layouts
        layout.addLayout(data_layout)
        layout.addWidget(self.train_button)
        layout.addWidget(self.training_progress)
        layout.addWidget(QLabel("Training Log:"))
        layout.addWidget(self.training_log)
        layout.addWidget(self.loss_plot)
        layout.addWidget(self.acc_plot)

        self.training_tab.setLayout(layout)

    def init_classification_tab(self):
        layout = QVBoxLayout()

        # Model loading status
        self.model_status = QLabel("Model Status: Not Loaded")
        self.model_status.setStyleSheet("color: red")

        # Load Model button
        load_model_button = QPushButton("Load Trained Model")
        load_model_button.clicked.connect(self.load_trained_model)

        # Image selection
        image_layout = QHBoxLayout()
        self.image_path_edit = QLineEdit()
        self.image_path_edit.setReadOnly(True)
        browse_image_button = QPushButton("Select Images")
        browse_image_button.clicked.connect(self.browse_images)
        image_layout.addWidget(QLabel("Images:"))
        image_layout.addWidget(self.image_path_edit)
        image_layout.addWidget(browse_image_button)

        # Classify button
        self.classify_button = QPushButton("Classify Images")
        self.classify_button.clicked.connect(self.classify_images)
        self.classify_button.setEnabled(False)

        # Progress bar
        self.classification_progress = QProgressBar()
        self.classification_progress.setValue(0)
        self.classification_progress.setVisible(False)

        # Log area
        self.classification_log = QTextEdit()
        self.classification_log.setReadOnly(True)

        # Confidence plot
        self.confidence_plot = pg.PlotWidget(title="Confidence Scores")
        self.confidence_plot.setLabel('left', 'Probability')
        self.confidence_plot.setLabel('bottom', 'Classes')
        self.confidence_plot.showGrid(x=True, y=True)
        self.confidence_bar = pg.BarGraphItem(x=[], height=[], width=0.6, brush='r')
        self.confidence_plot.addItem(self.confidence_bar)

        # Display image
        self.display_label = QLabel("Selected Image Preview")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setFixedSize(500, 500)

        # Arrange layouts
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.model_status)
        top_layout.addWidget(load_model_button)

        layout.addLayout(top_layout)
        layout.addLayout(image_layout)
        layout.addWidget(self.classify_button)
        layout.addWidget(self.classification_progress)
        layout.addWidget(QLabel("Classification Log:"))
        layout.addWidget(self.classification_log)
        layout.addWidget(self.confidence_plot)
        layout.addWidget(self.display_label)

        self.classification_tab.setLayout(layout)

    def browse_data_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Data Folder")
        if folder:
            self.data_path_edit.setText(folder)
            self.train_button.setEnabled(True)

    def start_training(self):
        image_dir = self.data_path_edit.text()
        metadata_file = os.path.join(image_dir, 'metadata.csv')
        if not os.path.exists(metadata_file):
            QMessageBox.warning(self, "Missing Metadata", "metadata.csv not found in the selected folder.")
            return

        # Disable the train button to prevent multiple trainings
        self.train_button.setEnabled(False)
        self.training_progress.setVisible(True)
        self.training_progress.setValue(0)
        self.training_log.clear()
        self.loss_curve.clear()
        self.val_loss_curve.clear()
        self.acc_curve.clear()
        self.val_acc_curve.clear()

        # Initialize training signals
        self.signals = WorkerSignals()
        self.signals.progress.connect(self.update_training_progress)
        self.signals.finished.connect(self.training_finished)
        self.signals.error.connect(self.training_error)
        self.signals.log.connect(self.append_training_log)

        # Start training in a separate thread
        self.trainer_thread = TrainerThread(image_dir, metadata_file, self.signals)
        self.trainer_thread.start()

        # Optionally, connect additional signals to capture training history
        # This would require modifying the TrainerThread to emit history data

    def update_training_progress(self, value):
        self.training_progress.setValue(value)

    def training_finished(self):
        self.train_button.setEnabled(True)
        self.training_progress.setVisible(False)
        self.training_log.append("Training completed successfully.")
        self.model_status.setText("Model Status: Trained")
        self.model_status.setStyleSheet("color: green")
        QMessageBox.information(self, "Training Complete", "The CNN model has been trained and saved.")

    def training_error(self, error_msg):
        self.train_button.setEnabled(True)
        self.training_progress.setVisible(False)
        self.training_log.append(f"Error: {error_msg}")
        QMessageBox.critical(self, "Training Error", f"An error occurred during training:\n{error_msg}")

    def append_training_log(self, message):
        self.training_log.append(message)

    def load_trained_model(self):
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(self, "Select Trained Model", "",
                                              "Keras Model Files (*.keras);;All Files (*)", options=options)
        if file:
            try:
                self.model = load_model(file)
                with open('label_encoder.pkl', 'rb') as f:
                    self.le = pickle.load(f)
                self.class_names = self.le.classes_
                self.model_status.setText(f"Model Status: Loaded ({os.path.basename(file)})")
                self.model_status.setStyleSheet("color: green")
                self.classify_button.setEnabled(True)
                QMessageBox.information(self, "Model Loaded", f"Successfully loaded model from {file}.")
            except Exception as e:
                QMessageBox.critical(self, "Loading Error", f"Failed to load model:\n{str(e)}")

    def browse_images(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, "Select Images for Classification", "",
                                                "Image Files (*.png *.jpg *.jpeg *.bmp)", options=options)
        if files:
            self.image_path_edit.setText("; ".join([os.path.basename(f) for f in files]))
            self.classify_button.setEnabled(True)
            self.selected_image_paths = files

    def classify_images(self):
        if not self.model or not self.le:
            QMessageBox.warning(self, "Model Not Loaded", "Please load a trained model before classification.")
            return

        image_paths = self.selected_image_paths
        if not image_paths:
            QMessageBox.information(self, "No Images Selected", "Please select images to classify.")
            return

        # Reset previous results
        self.classification_log.clear()
        self.confidence_plot.clear()
        self.display_label.clear()
        self.display_label.setText("Selected Image Preview")

        # Disable classify button and show progress
        self.classify_button.setEnabled(False)
        self.classification_progress.setVisible(True)
        self.classification_progress.setValue(0)

        # Initialize classification signals
        self.classify_signals = WorkerSignals()
        self.classify_signals.progress.connect(self.update_classification_progress)
        self.classify_signals.finished.connect(self.classification_finished)
        self.classify_signals.error.connect(self.classification_error)
        self.classify_signals.log.connect(self.append_classification_log)

        # Start classification in a separate thread
        self.classifier_thread = ClassifierThread(image_paths, self.model, self.le, self.classify_signals)
        self.classifier_thread.start()

    def update_classification_progress(self, value):
        self.classification_progress.setValue(value)

    def classification_finished(self, results):
        predictions, probabilities = results
        self.classification_progress.setVisible(False)
        self.classify_button.setEnabled(True)
        self.classification_log.append("Classification completed successfully.")

        # Display results for the first image
        if predictions:
            first_image = self.classifier_thread.image_paths[0]
            first_prediction = predictions[0]
            first_prob = probabilities[0]
            self.classification_log.append(f"First Image: {os.path.basename(first_image)}")
            self.classification_log.append(f"Prediction: {first_prediction} ({np.max(first_prob) * 100:.2f}%)")
            self.display_image(first_image)
            self.plot_confidence(first_prob)
            QMessageBox.information(self, "Classification Complete",
                                    f"Classified {len(predictions)} images.\nResults for the first image are displayed.")

    def classification_error(self, error_msg):
        self.classification_progress.setVisible(False)
        self.classify_button.setEnabled(True)
        self.classification_log.append(f"Error: {error_msg}")
        QMessageBox.critical(self, "Classification Error", f"An error occurred during classification:\n{error_msg}")

    def append_classification_log(self, message):
        self.classification_log.append(message)

    def display_image(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.classification_log.append(f"Failed to load image for display: {img_path}")
            return
        img = cv2.resize(img, (500, 500))
        # Convert to QImage
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)
        self.display_label.setPixmap(pixmap)

    def plot_confidence(self, prob_array):
        self.confidence_plot.clear()
        class_labels = self.class_names
        x = np.arange(len(class_labels))
        y = prob_array
        brush = pg.mkBrush(color=(255, 0, 0, 150))
        bar_graph = pg.BarGraphItem(x=x, height=y, width=0.6, brush=brush)
        self.confidence_plot.addItem(bar_graph)
        self.confidence_plot.getAxis('bottom').setTicks([list(zip(x, class_labels))])
        self.confidence_plot.setYRange(0, 1)


def main():
    app = QApplication(sys.argv)
    gui = PEMClassifierGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
