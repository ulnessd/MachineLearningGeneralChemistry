import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os
import time
import numpy as np
import pandas as pd
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk
from PIL import Image
from sklearn.model_selection import train_test_split


class MLGui:
    def __init__(self, root):
        self.root = root
        self.root.title("CNN Lab Explorer")

        # Main Frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left Frame for Buttons and Console
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Canvas for Plotting
        plot_frame = tk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Dataset folder selector
        self.folder_button = tk.Button(left_frame, text="Select Dataset Folder", command=self.select_folder)
        self.folder_button.pack(pady=10)

        self.folder_label = tk.Label(left_frame, text="No folder selected")
        self.folder_label.pack()

        # Model configuration dropdown
        self.config_label = tk.Label(left_frame, text="Select Model Configuration")
        self.config_label.pack()

        self.configurations = ["2 Layers, 10 Epochs", "3 Layers, 10 Epochs", "3 Layers, 15 Epochs",
                               "3 Layers, 20 Epochs", "4 Layers, 10 Epochs"]
        self.config_var = tk.StringVar(self.root)
        self.config_var.set(self.configurations[0])
        self.config_menu = tk.OptionMenu(left_frame, self.config_var, *self.configurations)
        self.config_menu.pack(pady=10)

        # Run model button
        self.run_button = tk.Button(left_frame, text="Run Model", command=self.run_model)
        self.run_button.pack(pady=10)

        # Console for updates (with scrollbar)
        self.console = scrolledtext.ScrolledText(left_frame, height=10, width=50, wrap=tk.WORD)
        self.console.pack(pady=10)

        # Table for displaying training results
        self.training_table = ttk.Treeview(left_frame, columns=("Config", "Time", "MAE"), show="headings")
        self.training_table.heading("Config", text="Configuration")
        self.training_table.heading("Time", text="Training Time (s)")
        self.training_table.heading("MAE", text="Mean Absolute Error")
        self.training_table.pack(pady=10)

        # Canvas for plots
        self.fit_canvas = tk.Canvas(plot_frame, width=400, height=300)
        self.fit_canvas.pack(side="top", fill=tk.BOTH, expand=True)

        self.residual_canvas = tk.Canvas(plot_frame, width=400, height=300)
        self.residual_canvas.pack(side="bottom", fill=tk.BOTH, expand=True)

    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.folder_label.config(text=folder_path)
            self.console.insert(tk.END, f"Dataset folder selected: {folder_path}\n")
            self.console.yview(tk.END)

            # Load the dataset (CSV file)
            csv_path = os.path.join(folder_path, 'ft_metadata.csv')
            if os.path.exists(csv_path):
                self.data = pd.read_csv(csv_path)
                self.console.insert(tk.END, f"CSV file loaded successfully.\n")
                self.console.yview(tk.END)
            else:
                messagebox.showwarning("No CSV File", "The selected folder does not contain a CSV file.")
                return

            # Load and process images
            images_dir = os.path.join(folder_path, 'ft_images')
            if os.path.exists(images_dir):
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
                self.console.insert(tk.END, f"Images loaded successfully.\n")
                self.console.yview(tk.END)
            else:
                messagebox.showwarning("No Image Folder",
                                       "The selected folder does not contain the required 'ft_images' subfolder.")
                self.data = None
                return

    def run_model(self):
        if self.data is None or self.images is None:
            messagebox.showwarning("No Data", "Please select a dataset folder first.")
            return

        config = self.config_var.get()
        self.console.insert(tk.END, f"Running model with configuration: {config}\n")
        self.root.update()

        # Parse model configuration
        layers_config, epochs = self.parse_config(config)

        # Train model
        time_taken, mae = self.train_model(layers_config, epochs)

        # Display results
        self.training_table.insert("", "end", values=(config, f"{time_taken:.2f}", f"{mae:.2f}"))

        # Plot model results
        self.plot_model_data()

    def parse_config(self, config):
        # Parse the selected configuration to get layers and epochs
        config_parts = config.split(", ")
        layers_num = int(config_parts[0].split(" ")[0])
        epochs = int(config_parts[1].split(" ")[0])
        layers_config = [32 * (2 ** i) for i in range(layers_num)]
        return layers_config, epochs

    def train_model(self, layers_config, epochs):
        # Use sklearn's train_test_split to split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.images, self.data['TotalEdgePoints'], test_size=0.2,
                                                            random_state=42)

        model = models.Sequential()
        model.add(layers.Conv2D(layers_config[0], (3, 3), activation='relu', input_shape=(128, 128, 2)))
        model.add(layers.MaxPooling2D((2, 2)))

        for num_filters in layers_config[1:]:
            model.add(layers.Conv2D(num_filters, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(1, activation='linear'))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

        start_time = time.time()
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
        elapsed_time = time.time() - start_time

        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        self.model = model
        return elapsed_time, test_mae

    def plot_model_data(self):
        if self.model is None:
            messagebox.showwarning("No Model", "Please train a model first.")
            return

        X_train, X_test, y_train, y_test = train_test_split(self.images, self.data['TotalEdgePoints'], test_size=0.2,
                                                            random_state=42)
        y_pred = self.model.predict(X_test)

        # Plot Actual vs Predicted
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        ax1.scatter(y_test, y_pred)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax1.set_xlabel('Actual Site Defects')
        ax1.set_ylabel('Predicted Site Defects')
        ax1.set_title('Model Configuration Evaluation')
        fig1.tight_layout()  # Automatically adjust spacing

        # Plot Residuals
        residuals = y_test - y_pred.flatten()
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.scatter(y_test, residuals)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Actual Site Defects')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residuals')
        fig2.tight_layout()  # Automatically adjust spacing

        # Clear the previous plots in the canvas
        for widget in self.fit_canvas.winfo_children():
            widget.destroy()
        for widget in self.residual_canvas.winfo_children():
            widget.destroy()

        # Display the Actual vs Predicted plot on the fit canvas
        fig1_canvas = FigureCanvasTkAgg(fig1, master=self.fit_canvas)
        fig1_canvas.draw()
        fig1_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Display the Residuals plot on the residual canvas
        fig2_canvas = FigureCanvasTkAgg(fig2, master=self.residual_canvas)
        fig2_canvas.draw()
        fig2_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = MLGui(root)
    root.mainloop()
