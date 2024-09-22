import time
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, explained_variance_score
from scipy.stats import pearsonr

class MLGui:
    def __init__(self, root):
        self.root = root
        self.root.title("Cobber Machine Learning Explorer")
        self.root.iconbitmap("CobberMLIcon.ico")  # Set the icon for the main window

        self.root.geometry("375x400")  # Example size: 800 pixels wide by 600 pixels high

        # Variables to store data and model
        self.data = None
        self.model = None

        # Create widgets
        self.create_widgets()

    def create_widgets(self):
        # File dialog to open CSV
        self.open_button = tk.Button(self.root, text="Open CSV", command=self.open_file)
        self.open_button.pack(pady=10)

        # ML Algorithm dropdown
        self.alg_label = tk.Label(self.root, text="Select ML Algorithm")
        self.alg_label.pack()

        self.algorithms = ["Linear Regression", "Decision Tree", "Random Forest", "Support Vector Machine",
                           "k-Nearest Neighbors"]
        self.alg_var = tk.StringVar(self.root)
        self.alg_var.set(self.algorithms[0])
        self.alg_menu = tk.OptionMenu(self.root, self.alg_var, *self.algorithms)
        self.alg_menu.pack(pady=10)

        # Train/Test Split Slider
        self.split_label = tk.Label(self.root, text="Train/Test Split (%)")
        self.split_label.pack()

        self.split_var = tk.DoubleVar(value=80)  # Default value of 80%
        self.split_slider = tk.Scale(self.root, from_=50, to=90, orient=tk.HORIZONTAL, variable=self.split_var)
        self.split_slider.pack(pady=10)

        # Button to run ML training
        self.train_button = tk.Button(self.root, text="Run ML Training", command=self.run_training)
        self.train_button.pack(pady=10)

        # Entry for manual predictions
        self.pred_label = tk.Label(self.root, text="Predict CORNYness")
        self.pred_label.pack()

        self.x1_entry = tk.Entry(self.root)
        self.x1_entry.pack(pady=5)
        self.x1_entry.insert(0, "Enter EARitability")

        self.x2_entry = tk.Entry(self.root)
        self.x2_entry.pack(pady=5)
        self.x2_entry.insert(0, "Enter aMAIZEingness")

        self.pred_button = tk.Button(self.root, text="Predict CORNYness", command=self.predict_y)
        self.pred_button.pack(pady=10)

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                messagebox.showinfo("File Loaded", "CSV file loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")


    def run_training(self):
        if self.data is not None:
            X = self.data[['EARitability', 'aMAIZEingness']]
            Y = self.data['CORNYness']
            train_size = self.split_var.get() / 100
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_size, random_state=42)

            # Select and train the model based on the user's choice
            if self.alg_var.get() == "Linear Regression":
                self.model = LinearRegression()
            elif self.alg_var.get() == "Decision Tree":
                self.model = DecisionTreeRegressor(random_state=42)
            elif self.alg_var.get() == "Random Forest":
                self.model = RandomForestRegressor(random_state=42)
            elif self.alg_var.get() == "Support Vector Machine":
                self.model = SVR()
            elif self.alg_var.get() == "k-Nearest Neighbors":
                self.model = KNeighborsRegressor()

            # Record the start time
            start_time = time.time()

            # Train the selected model
            self.model.fit(X_train, Y_train)

            # Record the end time and calculate training duration
            training_time = time.time() - start_time

            # Make predictions
            Y_pred = self.model.predict(X_test)

            # Calculate performance metrics
            mse = mean_squared_error(Y_test, Y_pred)
            rmse = mean_squared_error(Y_test, Y_pred, squared=False)
            mae = mean_absolute_error(Y_test, Y_pred)
            explained_var = explained_variance_score(Y_test, Y_pred)
            r2 = r2_score(Y_test, Y_pred)

            # Residuals analysis
            residuals = Y_test - Y_pred
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)

            # Pearson correlation coefficient
            corr_coef, _ = pearsonr(Y_test, Y_pred)

            # Display the results to the user
            messagebox.showinfo(
                "Training Complete",
                f"Model trained successfully!\n"
                f"Training Time: {training_time:.4f} seconds\n"
                f"MSE: {mse:.4f}\n"
                f"RMSE: {rmse:.4f}\n"
                f"MAE: {mae:.4f}\n"
                f"R²: {r2:.4f}\n"
                f"Explained Variance: {explained_var:.4f}\n"
                f"Pearson Correlation Coefficient: {corr_coef:.4f}\n"
                f"Mean Residual: {mean_residual:.4f}\n"
                f"Standard Deviation of Residuals: {std_residual:.4f}"
            )

            # Plot actual vs predicted values and residuals side by side
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))

            # Actual vs Predicted plot
            axs[0].scatter(Y_test, Y_pred)
            axs[0].plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')  # Red dashed line
            axs[0].set_xlabel("Actual CORNYness")
            axs[0].set_ylabel("Predicted CORNYness")
            axs[0].set_title(f"{self.alg_var.get()} - Actual vs Predicted")

            # Residuals plot
            axs[1].scatter(Y_test, residuals)
            axs[1].axhline(y=0, color='r', linestyle='--')  # Red dashed line at 0
            axs[1].set_xlabel("Actual CORNYness")
            axs[1].set_ylabel("Residuals")
            axs[1].set_title(f"Residuals Plot for {self.alg_var.get()}")

            plt.tight_layout()
            plt.show()
        else:
            messagebox.showwarning("No Data", "Please load a CSV file first.")

    def predict_y(self):
        if self.model is not None:
            try:
                x1 = float(self.x1_entry.get())
                x2 = float(self.x2_entry.get())
                y_pred = self.model.predict([[x1, x2]])
                messagebox.showinfo("Prediction", f"Predicted Y: {y_pred[0]}")
            except ValueError:
                messagebox.showerror("Input Error", "Please enter valid numbers for X1 and X2.")
            except Exception as e:
                messagebox.showerror("Prediction Error", f"Prediction failed: {str(e)}")
        else:
            messagebox.showwarning("No Model", "Please train a model first.")


if __name__ == "__main__":
    root = tk.Tk()
    app = MLGui(root)
    root.mainloop()
