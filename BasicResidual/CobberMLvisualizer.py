import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QHBoxLayout, QTextEdit
)
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class MLDataVisualizer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cobber Machine Learning Data Visualizer")
        self.setGeometry(100, 100, 1400, 900)  # Increased width for metrics display

        # Initialize layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Add Load Button
        self.load_button = QPushButton("Load CSV Dataset")
        self.load_button.clicked.connect(self.load_csv)
        self.layout.addWidget(self.load_button)

        # Add Info Label
        self.info_label = QLabel("Please load a CSV dataset to visualize.")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.info_label)

        # Create horizontal layout for plots and metrics
        self.main_content_layout = QHBoxLayout()
        self.layout.addLayout(self.main_content_layout)

        # Initialize Predicted vs Actual Plot
        self.pv_aplot = pg.PlotWidget(title="Predicted vs Actual Value")
        self.pv_aplot.setLabel('left', 'Predicted')
        self.pv_aplot.setLabel('bottom', 'Actual')
        self.pv_aplot.addLegend()
        self.pv_aplot.showGrid(x=True, y=True)
        self.main_content_layout.addWidget(self.pv_aplot)

        # Initialize Residuals Plot
        self.residuals_plot = pg.PlotWidget(title="Residuals (Predicted - Actual)")
        self.residuals_plot.setLabel('left', 'Residual')
        self.residuals_plot.setLabel('bottom', 'Actual')
        self.residuals_plot.addLegend()
        self.residuals_plot.showGrid(x=True, y=True)
        self.main_content_layout.addWidget(self.residuals_plot)

        # Initialize Metrics Display
        self.metrics_layout = QVBoxLayout()
        self.main_content_layout.addLayout(self.metrics_layout)

        self.metrics_label = QLabel("Model Evaluation Metrics:")
        self.metrics_label.setAlignment(Qt.AlignTop)
        self.metrics_layout.addWidget(self.metrics_label)

        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_layout.addWidget(self.metrics_text)

        # Initialize Data
        self.data = None

    def load_csv(self):
        # Open file dialog to select CSV
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open CSV Dataset", "", "CSV Files (*.csv);;All Files (*)", options=options
        )
        if file_name:
            try:
                self.data = pd.read_csv(file_name)
                if 'Actual' not in self.data.columns or 'Predicted' not in self.data.columns:
                    self.info_label.setText("Error: CSV must contain 'Actual' and 'Predicted' columns.")
                    return
                self.info_label.setText(f"Loaded dataset: {file_name}")
                self.plot_data()
            except Exception as e:
                self.info_label.setText(f"Error loading CSV: {e}")

    def plot_data(self):
        if self.data is None:
            return

        actual = self.data['Actual'].values
        predicted = self.data['Predicted'].values
        residuals = predicted - actual

        # Compute Metrics
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        r2 = r2_score(actual, predicted)

        # Update Metrics Display
        metrics_text = f"Mean Absolute Error (MAE): {mae:.2f}\n" \
                       f"Mean Squared Error (MSE): {mse:.2f}\n" \
                       f"R-squared (R²): {r2:.2f}"
        self.metrics_text.setText(metrics_text)

        # Clear previous plots
        self.pv_aplot.clear()
        self.residuals_plot.clear()

        # Plot Predicted vs Actual
        scatter = pg.ScatterPlotItem(actual, predicted, pen=pg.mkPen(None), brush=pg.mkBrush(100, 100, 255, 150),
                                     size=5, name="Data Points")
        self.pv_aplot.addItem(scatter)

        # Plot 45-degree line
        min_val = min(np.min(actual), np.min(predicted))
        max_val = max(np.max(actual), np.max(predicted))
        self.pv_aplot.plot([min_val, max_val], [min_val, max_val], pen=pg.mkPen('r', width=2), name="45-Degree Line")

        # Plot Residuals
        residual_scatter = pg.ScatterPlotItem(actual, residuals, pen=pg.mkPen(None),
                                              brush=pg.mkBrush(255, 100, 100, 150), size=5, name="Residuals")
        self.residuals_plot.addItem(residual_scatter)

        # Plot Zero Line for Residuals
        self.residuals_plot.plot([np.min(actual), np.max(actual)], [0, 0], pen=pg.mkPen('r', width=2), name="Zero Line")

        # Adjust plot ranges
        self.pv_aplot.setXRange(min_val - 5, max_val + 5)
        self.pv_aplot.setYRange(min_val - 5, max_val + 5)

        # For Residuals Plot
        res_min = np.min(residuals)
        res_max = np.max(residuals)
        self.residuals_plot.setXRange(np.min(actual) - 5, np.max(actual) + 5)
        self.residuals_plot.setYRange(res_min - 5, res_max + 5)

        # Refresh the plots
        self.pv_aplot.repaint()
        self.residuals_plot.repaint()


def main():
    app = QApplication(sys.argv)
    window = MLDataVisualizer()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
