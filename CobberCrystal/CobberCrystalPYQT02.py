import sys
import os
import time
import numpy as np
import pandas as pd
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from PIL import Image

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel, QVBoxLayout,
    QHBoxLayout, QFileDialog, QMessageBox, QTextEdit, QComboBox, QTableWidget,
    QTableWidgetItem, QSizePolicy, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QTextCursor

import pyqtgraph as pg
from tensorflow.keras.callbacks import Callback

class TrainingProgressBar(Callback):
    """
    Custom TensorFlow callback to emit progress updates.
    """
    def __init__(self, emit_progress_func, total_epochs):
        super().__init__()
        self.emit_progress = emit_progress_func
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        # Calculate progress percentage
        progress = int((epoch + 1) / self.total_epochs * 100)
        self.emit_progress(progress)  # Correct invocation

class TrainModelThread(QThread):
    """
    QThread subclass to handle model training in a separate thread.
    """
    message = pyqtSignal(str)          # Signal for console messages
    progress = pyqtSignal(int)         # Signal for progress bar updates
    result = pyqtSignal(float, float)  # Signal for training results

    def __init__(self, images, targets, layers_config, epochs):
        super().__init__()
        self.images = images
        self.targets = targets
        self.layers_config = layers_config
        self.epochs = epochs
        self.model = None

    def run(self):
        self.message.emit("Starting model training...")

        # Split the data
        self.message.emit("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            self.images, self.targets, test_size=0.2, random_state=42
        )

        # Build the model
        self.message.emit("Building the CNN model...")
        model = models.Sequential()
        model.add(layers.Conv2D(self.layers_config[0], (3, 3), activation='relu', input_shape=(128, 128, 2)))
        model.add(layers.MaxPooling2D((2, 2)))

        for num_filters in self.layers_config[1:]:
            model.add(layers.Conv2D(num_filters, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(1, activation='linear'))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

        # Define the emit_progress function
        def emit_progress(progress):
            self.progress.emit(progress)

        # Train the model with the progress callback
        self.message.emit("Training the model...")
        start_time = time.time()
        model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=32,
            verbose=0,
            callbacks=[TrainingProgressBar(emit_progress, self.epochs)]
        )
        elapsed_time = time.time() - start_time
        self.message.emit(f"Training completed in {elapsed_time:.2f} seconds.")

        # Evaluate the model
        self.message.emit("Evaluating the model...")
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        self.message.emit(f"Model evaluation completed. MAE: {test_mae:.2f}")

        self.model = model
        self.result.emit(elapsed_time, test_mae)

class MainWindow(QMainWindow):
    """
    Main window of the application.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CNN Lab Explorer")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize data attributes
        self.data = None
        self.images = None
        self.model = None

        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Main layout
        main_layout = QHBoxLayout()
        self.central_widget.setLayout(main_layout)

        # Left panel for controls and console
        left_panel = QVBoxLayout()
        main_layout.addLayout(left_panel, 1)

        # Select Dataset Folder Button
        self.folder_button = QPushButton("Select Dataset Folder")
        self.folder_button.clicked.connect(self.select_folder)
        left_panel.addWidget(self.folder_button)

        # Selected Folder Label
        self.folder_label = QLabel("No folder selected")
        left_panel.addWidget(self.folder_label)

        # Model Configuration Dropdown
        self.config_label = QLabel("Select Model Configuration")
        left_panel.addWidget(self.config_label)

        self.configurations = [
            "2 Layers, 10 Epochs",
            "3 Layers, 10 Epochs",
            "3 Layers, 15 Epochs",
            "3 Layers, 20 Epochs",
            "4 Layers, 10 Epochs"
        ]
        self.config_combo = QComboBox()
        self.config_combo.addItems(self.configurations)
        left_panel.addWidget(self.config_combo)

        # Run Model Button
        self.run_button = QPushButton("Run Model")
        self.run_button.clicked.connect(self.run_model)
        self.run_button.setEnabled(False)
        left_panel.addWidget(self.run_button)

        # Console Output
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        left_panel.addWidget(self.console)

        # Training Results Table
        self.training_table = QTableWidget()
        self.training_table.setColumnCount(3)
        self.training_table.setHorizontalHeaderLabels(["Configuration", "Training Time (s)", "MAE"])
        self.training_table.horizontalHeader().setStretchLastSection(True)
        left_panel.addWidget(self.training_table)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Training Progress: %p%")
        left_panel.addWidget(self.progress_bar)

        # Right panel for plots
        right_panel = QVBoxLayout()
        main_layout.addLayout(right_panel, 3)

        # Actual vs Predicted Plot
        self.plot_actual_predicted = pg.PlotWidget(title="Actual vs Predicted Surface Reactivity")
        self.plot_actual_predicted.setLabel('left', 'Predicted Site Defects')
        self.plot_actual_predicted.setLabel('bottom', 'Actual Site Defects')
        self.plot_actual_predicted.showGrid(x=True, y=True)
        right_panel.addWidget(self.plot_actual_predicted)

        # Residuals Plot
        self.plot_residuals = pg.PlotWidget(title="Residuals")
        self.plot_residuals.setLabel('left', 'Residuals')
        self.plot_residuals.setLabel('bottom', 'Actual Site Defects')
        self.plot_residuals.showGrid(x=True, y=True)
        right_panel.addWidget(self.plot_residuals)

        # Initialize threads
        self.train_thread = None

    def select_folder(self):
        """
        Open a dialog to select the dataset folder and load data.
        """
        folder_path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder", "")
        if folder_path:
            self.folder_label.setText(folder_path)
            self.append_console(f"Dataset folder selected: {folder_path}")
            self.append_console("Loading data set...")
            QApplication.processEvents()  # Ensure the message is displayed before loading

            # Load the dataset (CSV file)
            csv_path = os.path.join(folder_path, 'ft_metadata.csv')
            if os.path.exists(csv_path):
                try:
                    self.data = pd.read_csv(csv_path)
                    self.append_console("CSV file loaded successfully.")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to load CSV file: {e}")
                    self.data = None
                    return
            else:
                QMessageBox.warning(self, "No CSV File", "The selected folder does not contain a 'ft_metadata.csv' file.")
                self.data = None
                return

            # Load and process images
            images_dir = os.path.join(folder_path, 'ft_images')
            if os.path.exists(images_dir):
                try:
                    self.images = []
                    for i, row in self.data.iterrows():
                        full_image_path = os.path.join(images_dir, row['FullFTFilename'])
                        zoomed_image_path = os.path.join(images_dir, row['ZoomedFTFilename'])

                        full_image = Image.open(full_image_path).convert('L')  # Convert to grayscale
                        zoomed_image = Image.open(zoomed_image_path).convert('L')

                        # Resize images to 128x128
                        full_image = full_image.resize((128, 128))
                        zoomed_image = zoomed_image.resize((128, 128))

                        # Combine full and zoomed images into a single array with shape (128, 128, 2)
                        combined_image = np.stack([np.array(full_image), np.array(zoomed_image)], axis=-1)

                        self.images.append(combined_image)

                    self.images = np.array(self.images)
                    self.images = self.images / 255.0  # Normalize the images
                    self.append_console("Images loaded and processed successfully.")
                    self.run_button.setEnabled(True)
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to load images: {e}")
                    self.images = None
                    return
            else:
                QMessageBox.warning(self, "No Image Folder",
                                    "The selected folder does not contain the required 'ft_images' subfolder.")
                self.data = None
                self.images = None
                return

    def run_model(self):
        """
        Start the model training in a separate thread.
        """
        if self.data is None or self.images is None:
            QMessageBox.warning(self, "No Data", "Please select a dataset folder first.")
            return

        config = self.config_combo.currentText()
        self.append_console(f"Running model with configuration: {config}")
        self.run_button.setEnabled(False)
        self.config_combo.setEnabled(False)

        # Parse model configuration
        layers_config, epochs = self.parse_config(config)

        # Start training in a separate thread
        self.train_thread = TrainModelThread(
            images=self.images,
            targets=self.data['TotalEdgePoints'].values,
            layers_config=layers_config,
            epochs=epochs
        )
        self.train_thread.message.connect(self.append_console)
        self.train_thread.progress.connect(self.update_progress_bar)
        self.train_thread.result.connect(self.training_finished)
        self.train_thread.start()

    def parse_config(self, config):
        """
        Parse the selected configuration to get layers and epochs.
        """
        config_parts = config.split(", ")
        layers_num = int(config_parts[0].split(" ")[0])
        epochs = int(config_parts[1].split(" ")[0])
        layers_config = [32 * (2 ** i) for i in range(layers_num)]
        return layers_config, epochs

    def training_finished(self, time_taken, mae):
        """
        Handle the results after training is finished.
        """
        config = self.config_combo.currentText()
        row_position = self.training_table.rowCount()
        self.training_table.insertRow(row_position)
        self.training_table.setItem(row_position, 0, QTableWidgetItem(config))
        self.training_table.setItem(row_position, 1, QTableWidgetItem(f"{time_taken:.2f}"))
        self.training_table.setItem(row_position, 2, QTableWidgetItem(f"{mae:.2f}"))

        self.model = self.train_thread.model
        self.append_console("Training thread finished.")
        self.run_button.setEnabled(True)
        self.config_combo.setEnabled(True)

        # Reset progress bar to 100% if not already
        self.progress_bar.setValue(100)

        # Plot the results
        self.plot_model_data()

    def append_console(self, text):
        """
        Append text to the console and scroll to the end.
        """
        self.console.append(text)
        self.console.moveCursor(QTextCursor.End)

    def update_progress_bar(self, value):
        """
        Update the progress bar with the given value.
        """
        self.progress_bar.setValue(value)

    def plot_model_data(self):
        """
        Generate and display the plots after training.
        """
        if self.model is None:
            QMessageBox.warning(self, "No Model", "Please train a model first.")
            return

        self.append_console("Generating plots...")

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.images, self.data['TotalEdgePoints'].values, test_size=0.2, random_state=42
        )
        y_pred = self.model.predict(X_test).flatten()

        # Clear previous plots
        self.plot_actual_predicted.clear()
        self.plot_residuals.clear()

        # Actual vs Predicted Plot
        scatter = pg.ScatterPlotItem(x=y_test, y=y_pred, pen=pg.mkPen(None), brush=pg.mkBrush(100, 100, 255, 120))
        self.plot_actual_predicted.addItem(scatter)
        # Diagonal line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        self.plot_actual_predicted.plot([min_val, max_val], [min_val, max_val], pen=pg.mkPen('r', width=2, style=Qt.DashLine))

        self.plot_actual_predicted.setXRange(min_val, max_val)
        self.plot_actual_predicted.setYRange(min_val, max_val)

        # Residuals Plot
        residuals = y_pred - y_test
        scatter_res = pg.ScatterPlotItem(x=y_test, y=residuals, pen=pg.mkPen(None), brush=pg.mkBrush(255, 100, 100, 120))
        self.plot_residuals.addItem(scatter_res)
        # Horizontal line at y=0
        self.plot_residuals.plot([min_val, max_val], [0, 0], pen=pg.mkPen('r', width=2, style=Qt.DashLine))

        self.plot_residuals.setXRange(min_val, max_val)
        residual_min = min(residuals)
        residual_max = max(residuals)
        self.plot_residuals.setYRange(residual_min, residual_max)

        self.append_console("Plots updated successfully.")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
