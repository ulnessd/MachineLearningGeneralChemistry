import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math
import threading
import time

class ChemiInformaticsLab(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ChemiInformatics Lab")
        self.geometry("1400x700")  # Adjusted window size to accommodate all elements
        self.resizable(False, False)

        # Initialize protein
        self.protein_segments = 100  # Number of protein segments
        self.protein_values = np.array([math.cos(2.7 * x / self.protein_segments)**2 for x in range(self.protein_segments)])
        self.protein_values = self.protein_values / np.max(self.protein_values)  # Normalize between 0 and 1

        # Colormap
        self.colormap = plt.get_cmap('rainbow')

        # Ligands list: list of tuples (ligand_values, energy)
        self.ligands = []
        self.top_ligands = []  # Top three ligands

        # Generation Tracking
        self.current_generation = 0
        self.generations = []      # List to store generation numbers
        self.best_energies = []    # List to store best energies per generation
        self.current_top_ligand = None  # To store the top ligand for next generation

        # Setup UI
        self.create_widgets()

    def create_widgets(self):
        # Main Frame
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left Frame for Graph and Protein/Ligand Display
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # Matplotlib Figure for Energy vs Generation Graph
        self.figure, self.ax = plt.subplots(figsize=(6, 3))
        self.ax.set_title("Best Match Energy vs Generation")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Energy")
        self.ax.grid(True)

        #self.energy_line, = self.ax.plot([], [], 'bo-', label="Best Energy")
        #self.ax.legend()
        self.figure.tight_layout()

        # Embed the matplotlib figure into Tkinter
        self.canvas_fig = FigureCanvasTkAgg(self.figure, master=left_frame)
        self.canvas_fig.draw()
        self.canvas_fig.get_tk_widget().pack(pady=10)

        # Canvas for Protein and Ligands
        # Reduced height to prevent squeezing out other UI elements
        self.canvas = tk.Canvas(left_frame, width=1100, height=200, bg="white")
        self.canvas.pack(pady=10)

        # Right Frame for Controls and Top Ligands
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        # Present Ligand Button
        self.present_btn = tk.Button(right_frame, text="Present Ligand", command=self.present_ligand, width=25, height=3)
        self.present_btn.pack(pady=10)

        # Run Seed Generation Button
        self.seed_btn = tk.Button(right_frame, text="Run Seed Generation", command=self.run_seed_generation, width=25, height=3)
        self.seed_btn.pack(pady=10)

        # Run Next Generation Button
        self.next_gen_btn = tk.Button(right_frame, text="Run Next Generation", command=self.run_next_generation, width=25, height=3)
        self.next_gen_btn.pack(pady=10)

        # Ligands per generation input
        ligands_frame = tk.Frame(right_frame)
        ligands_frame.pack(pady=10)
        tk.Label(ligands_frame, text="Ligands per Gen:", font=("Arial", 12)).pack(side=tk.LEFT)
        self.ligands_input = tk.Entry(ligands_frame, width=10, font=("Arial", 12))
        self.ligands_input.insert(0, "100")
        self.ligands_input.pack(side=tk.LEFT, padx=5)

        # Perturbation level input
        perturb_frame = tk.Frame(right_frame)
        perturb_frame.pack(pady=10)
        tk.Label(perturb_frame, text="Perturbation Level:", font=("Arial", 12)).pack(side=tk.LEFT)
        self.perturb_input = tk.Entry(perturb_frame, width=10, font=("Arial", 12))
        self.perturb_input.insert(0, "0.05")
        self.perturb_input.pack(side=tk.LEFT, padx=5)

        # Generation Label
        self.generation_label = tk.Label(right_frame, text="Generation: 0", font=("Arial", 14))
        self.generation_label.pack(pady=20)

        # Top Ligands Display
        top_ligands_label = tk.Label(right_frame, text="Top 3 Ligands:", font=("Arial", 12, "bold"))
        top_ligands_label.pack(pady=10)

        self.top_ligands_canvases = []
        for i in range(3):
            canvas = tk.Canvas(right_frame, width=200, height=60, bg="white", bd=1, relief=tk.SUNKEN)
            canvas.pack(pady=5)
            self.top_ligands_canvases.append(canvas)

        # Initial Draw
        self.draw_protein()

    def draw_protein(self):
        self.canvas.delete("all")
        width = 1100
        height = 300
        x_start = 50
        y_start = 150  # Y-coordinate where protein is drawn
        rect_width = width - 100  # Adjusted width to fit within canvas
        rect_height = 25  # Control the height of the protein rectangles here
        segment_width = rect_width / self.protein_segments

        for i in range(self.protein_segments):
            val = self.protein_values[i]
            color = self.get_color(val)
            x0 = x_start + i * segment_width
            y0 = y_start
            x1 = x0 + segment_width
            y1 = y0 + rect_height
            self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")

        # Label
        self.canvas.create_text(x_start + rect_width / 2, y_start - 10, text="Protein Binding Site", font=("Arial", 12))

    def get_color(self, value):
        # value is between 0 and 1
        rgba = self.colormap(value)
        rgb = tuple(int(255 * c) for c in rgba[:3])
        return '#%02x%02x%02x' % rgb

    def present_ligand(self, ligand_values=None):
        if ligand_values is None:
            # Generate random ligand
            ligand_values = np.random.rand(self.protein_segments)
        else:
            ligand_values = np.clip(ligand_values, 0, 1)  # Ensure within [0,1]

        # Calculate RMSD
        rms = np.sqrt(np.mean((self.protein_values - ligand_values)**2))

        # Calculate binding energy
        try:
            ratio = (.5 - rms) / rms
            if ratio <= 0:
                E = 20  # Avoid log of non-positive number
            else:
                E = -5 * math.log(ratio)
        except Exception as e:
            print(f"Error calculating binding energy: {e}")
            E = 0  # Handle division by zero or log issues

        # Determine color based on E
        if E > 1:
            color = 'red'
        elif E < -1:
            color = 'green'
        else:
            color = 'blue'

        # Add to ligands list
        self.ligands.append((ligand_values, E))

        # Update top ligands
        self.update_top_ligands()

        # Update canvas
        self.update_canvas(ligand_values, E, color)

    def update_top_ligands(self):
        # Sort ligands based on energy (ascending order since lower energy is better)
        sorted_ligands = sorted(self.ligands, key=lambda x: x[1])
        self.top_ligands = sorted_ligands[:3]
        if self.top_ligands:
            self.current_top_ligand = self.top_ligands[0][0]

        # Update top ligands display
        for idx, canvas in enumerate(self.top_ligands_canvases):
            if idx < len(self.top_ligands):
                ligand, energy = self.top_ligands[idx]
                canvas.delete("all")
                width = 200
                height = 60  # **Increased height for top ligand boxes**
                rect_width = width
                rect_height = 20  # **Control the height of the top ligand rectangles here**
                segment_width = rect_width / self.protein_segments
                for i in range(self.protein_segments):
                    val = ligand[i]
                    color = self.get_color(val)
                    x0 = i * segment_width
                    y0 = (height - rect_height) / 2 - 10  # Adjusted y0 for better text placement
                    x1 = x0 + segment_width
                    y1 = y0 + rect_height
                    canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")
                # Energy Text
                if energy > 0:
                    energy_text = f"E={energy:.2f}"
                else:
                    energy_text = f"E={energy:.2f}"
                canvas.create_text(width / 2, height - 10, text=energy_text, font=("Arial", 12))
            else:
                canvas.delete("all")

            # Update the graph with the latest best energy
        if self.current_generation > 0 and self.top_ligands:
            best_energy = self.top_ligands[0][1]
            self.generations.append(self.current_generation)
            self.best_energies.append(best_energy)
            self.ax.plot(self.current_generation, best_energy, 'ro')  # Plot the new point



            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas_fig.draw()
    def update_canvas(self, ligand_values, energy, color):
        # Clear previous ligand
        self.canvas.delete("ligand")

        # Draw ligand in a fixed position
        width = 1100
        height = 300
        x_start = 50
        y_start = 50  # Fixed position above the protein
        rect_width = width - 100  # Adjusted width to fit within canvas
        rect_height = 25  # Height of the ligand rectangles
        segment_width = rect_width / self.protein_segments

        for i in range(self.protein_segments):
            val = ligand_values[i]
            ligand_color = self.get_color(val)
            x0 = x_start + i * segment_width
            y0 = y_start
            x1 = x0 + segment_width
            y1 = y0 + rect_height
            # Tag 'ligand' to identify these rectangles for easy deletion
            self.canvas.create_rectangle(x0, y0, x1, y1, fill=ligand_color, outline="", tags="ligand")

        # Energy Text
        text_color = color
        # Position the energy text above the ligand rectangle
        self.canvas.create_text(x_start + rect_width / 2, y_start - 10,
                                text=f"E={energy:.2f}", fill=text_color, font=("Arial", 12), tags="ligand")

    def run_seed_generation(self):
        # Clear previous ligands
        self.ligands = []
        self.top_ligands = []
        self.current_generation = 0
        self.generations = []
        self.best_energies = []
        self.generation_label.config(text=f"Generation: {self.current_generation}")
        self.canvas.delete("all")
        self.draw_protein()
        for canvas in self.top_ligands_canvases:
            canvas.delete("all")

        # Clear the graph
        self.ax.cla()
        self.ax.set_title("Best Match Energy vs Generation")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Energy")
        self.ax.grid(True)
        #self.energy_line, = self.ax.plot([], [], 'bo-')
        #self.ax.legend()
        self.figure.tight_layout()
        self.canvas_fig.draw()

        # Get number of ligands
        try:
            num_ligands = int(self.ligands_input.get())
            if num_ligands <= 0:
                raise ValueError
        except:
            messagebox.showerror("Invalid Input", "Please enter a valid positive integer for ligands per generation.")
            return

        # Disable buttons during generation
        self.disable_buttons()

        # Start generation in a separate thread to avoid freezing the GUI
        threading.Thread(target=self.seed_generation_thread, args=(num_ligands,), daemon=True).start()

    def seed_generation_thread(self, num_ligands):
        for _ in range(num_ligands):
            self.present_ligand()
            time.sleep(0.01)  # Reduced delay for faster generation

        # Re-enable buttons after generation
        self.enable_buttons()

    def run_next_generation(self):
        if not self.top_ligands:
            messagebox.showwarning("No Ligands", "Please run seed generation first.")
            return

        # Get number of ligands and perturbation level
        try:
            num_ligands = int(self.ligands_input.get())
            if num_ligands <= 0:
                raise ValueError
        except:
            messagebox.showerror("Invalid Input", "Please enter a valid positive integer for ligands per generation.")
            return

        try:
            perturb = float(self.perturb_input.get())
            if perturb < 0:
                raise ValueError
        except:
            messagebox.showerror("Invalid Input", "Please enter a valid non-negative number for perturbation level.")
            return

        if self.current_top_ligand is None:
            messagebox.showwarning("No Top Ligand", "No top ligand available for next generation.")
            return

        # Update generation
        self.current_generation += 1
        self.generation_label.config(text=f"Generation: {self.current_generation}")

        # Disable buttons during generation
        self.disable_buttons()

        # Start generation in a separate thread to avoid freezing the GUI
        threading.Thread(target=self.next_generation_thread, args=(num_ligands, perturb), daemon=True).start()

    def next_generation_thread(self, num_ligands, perturb):
        for _ in range(num_ligands):
            perturbation = np.random.uniform(-perturb, perturb, self.protein_segments)
            new_ligand = self.current_top_ligand + perturbation
            new_ligand = np.clip(new_ligand, 0, 1)
            self.present_ligand(new_ligand)
            time.sleep(0.01)  # Reduced delay for faster generation

        # Re-enable buttons after generation
        self.enable_buttons()

    def disable_buttons(self):
        self.present_btn.config(state=tk.DISABLED)
        self.seed_btn.config(state=tk.DISABLED)
        self.next_gen_btn.config(state=tk.DISABLED)

    def enable_buttons(self):
        self.present_btn.config(state=tk.NORMAL)
        self.seed_btn.config(state=tk.NORMAL)
        self.next_gen_btn.config(state=tk.NORMAL)

if __name__ == "__main__":
    app = ChemiInformaticsLab()
    app.mainloop()
