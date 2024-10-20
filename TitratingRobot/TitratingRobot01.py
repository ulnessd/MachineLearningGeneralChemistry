import sys
import numpy as np
import gym
from gym import spaces
import random
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import matplotlib.pyplot as plt
import time





from PyQt5.QtWidgets import (
    QApplication, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QSpinBox, QFileDialog, QTextEdit, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ----------------------------
# Environment Definition
# ----------------------------

class RobotTrackEnv(gym.Env):
    """
    Custom Environment for a Titrating Robot performing titrations.
    Maps the original RobotTrackEnv to the titration scenario.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(RobotTrackEnv, self).__init__()
        # Dynamics constraints
        self.max_velocity = 1.25  # mL/s (flow rate)
        self.min_velocity = 0.0    # mL/s
        self.max_acceleration = 0.5  # mL/s² (turning on buret)
        self.min_acceleration = -0.5  # mL/s² (turning off buret)
        self.dt = 0.1  # time step in seconds

        # Action space: discrete accelerations {-0.5, 0, +0.5} mL/s²
        self.acceleration_values = [-0.5, 0.0, 0.5]  # mL/s²
        self.action_space = spaces.Discrete(len(self.acceleration_values))

        # Observation space: [position (mL), velocity (mL/s), warning_signal, d_estimated (mL)]
        low_obs = np.array([0.0, self.min_velocity, 0.0, 9.5], dtype=np.float32)
        high_obs = np.array([np.inf, self.max_velocity, 1.0, 24.5], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # Initialize the environment
        self.reset()

    def reset(self):
        """
        Reset the environment to an initial state and return the initial observation.
        """
        # Only randomize d_true and d_estimated if they haven't been set externally
        if not getattr(self, 'equivalence_point_set', False):
            self.d_true = np.random.uniform(10.0, 24.0)  # Equivalence point
            self.d_estimated = self.d_true + np.random.uniform(-0.5, 0.5)  # mL
        else:
            # Reset the flag after using the externally set d_true
            self.equivalence_point_set = False

        self.position = 0.0  # mL of titrant added
        self.velocity = 0.0  # mL/s
        self.done = False
        self.total_time = 0.0
        self.warning = 0.0
        self.steps = 0
        return self._get_state()

    def step(self, action):
        """
        Execute one time step within the environment.
        """
        if self.done:
            raise Exception("Episode has ended. Please reset the environment.")

        # Map action to acceleration
        acceleration = self.acceleration_values[action]  # mL/s²

        # Update velocity
        self.velocity += acceleration * self.dt
        self.velocity = np.clip(self.velocity, self.min_velocity, self.max_velocity)

        # Update position
        self.position += self.velocity * self.dt  # mL

        # Update total time
        self.total_time += self.dt

        # Calculate warning signal with adjusted function
        self.warning = self._calculate_warning()

        # Initialize reward
        reward = self.velocity * self.dt - 0.1  # Per-step reward

        # Check for early stopping
        if self.velocity == 0.0 and self.warning == 0.0 and self.position < self.d_true - 0.5:
            self.done = True
            reward += -50.0  # Severe penalty for stopping early
            return self._get_state(), reward, self.done, {}

        # Check for success
        if self.d_true - 0.1 <= self.position < self.d_true:
            self.done = True
            # Adjusted final reward without d_true dependency
            reward += 100.0 / self.total_time  # Encourages faster completion
        # Check if fallen off the cliff
        elif self.position > self.d_true:
            self.done = True
            reward += -100.0  # Severe penalty for failure

        # Increment steps
        self.steps += 1

        # Get the next state
        state = self._get_state()
        return state, reward, self.done, {}

    def _calculate_warning(self):
        """
        Calculate the warning signal based on current position and estimated equivalence point.
        W = Θ(x - d + 0.4) * (1 - Exp[-20*(x - d + 0.4)^2])
        """
        x = self.position
        d_estimated = self.d_estimated  # Use estimated d

        if x >= d_estimated - 0.40:
            exponent = -20 * (x - d_estimated + 0.4) ** 2
            w = 1 - np.exp(exponent)
            return np.clip(w, 0.0, 1.0)
        else:
            return 0.0

    def _get_state(self):
        """
        Return the current state as an observation.
        """
        return np.array([self.position, self.velocity, self.warning, self.d_estimated], dtype=np.float32)

    def render(self, mode='human'):
        """
        Render the environment. Prints the current state.
        """
        print(
            f"Position: {self.position:.2f} mL, Velocity: {self.velocity:.2f} mL/s, Warning: {self.warning:.2f}, d_estimated: {self.d_estimated:.2f} mL")

    def set_equivalence_point(self, d_true):
        """
        Set the true equivalence point and calculate the estimated equivalence point.
        """
        self.d_true = d_true
        self.d_estimated = self.d_true + np.random.uniform(-0.5, 0.5)  # mL
        self.equivalence_point_set = True  # Flag to indicate external setting


# ----------------------------
# DQN Model Definition
# ----------------------------

class DQN(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu', input_shape=(state_size,))
        self.dense2 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(action_size)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)

# ----------------------------
# Replay Memory Definition
# ----------------------------

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        # Unpack batch
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

# ----------------------------
# Training Thread Definition
# ----------------------------

class TrainingThread(QThread):
    update_log = pyqtSignal(str)
    update_animation = pyqtSignal(dict)  # Emits frame data for animation

    def __init__(self, num_episodes):
        super().__init__()
        self.num_episodes = num_episodes
        self.env = RobotTrackEnv()
        self.model = None  # To store the trained model
        self.target_model = None
        self.memory = ReplayMemory(10000)
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 1500
        self.batch_size = 64
        self.target_update = 10  # Update target network every 10 episodes
        self.steps_done = 0  # Initialize steps_done
        self.epsilon = self.epsilon_start  # Current epsilon

    def select_action(self, state, action_size):
        # Calculate epsilon using exponential decay
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.epsilon = epsilon  # Correctly assign the decayed epsilon
        self.steps_done += 1
        if random.random() < self.epsilon:
            return random.randrange(action_size)
        else:
            state = np.array([state], dtype=np.float32)
            q_values = self.model(state)
            return np.argmax(q_values[0])

    def run(self):
        try:
            # Initialize model and target model
            state_size = self.env.observation_space.shape[0]
            action_size = self.env.action_space.n
            self.model = DQN(state_size, action_size)
            # Build the model by passing a dummy input
            dummy_input = tf.constant([[0.0, 0.0, 0.0, 0.0]], dtype=tf.float32)
            self.model(dummy_input)
            self.target_model = DQN(state_size, action_size)
            self.target_model(dummy_input)
            self.target_model.set_weights(self.model.get_weights())

            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

            for episode in range(1, self.num_episodes + 1):
                state = self.env.reset()
                total_reward = 0
                positions = [self.env.position]
                for t in range(1, 1501):  # Max steps per episode
                    # Epsilon-greedy action selection
                    action = self.select_action(state, action_size)

                    next_state, reward, done, _ = self.env.step(action)
                    total_reward += reward

                    # Store transition in replay memory
                    self.memory.push(state, action, reward, next_state, done)
                    state = next_state
                    positions.append(self.env.position)

                    # Perform experience replay
                    if len(self.memory) >= self.batch_size:
                        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
                        states = tf.convert_to_tensor(states, dtype=tf.float32)
                        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
                        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
                        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
                        actions = tf.convert_to_tensor(actions, dtype=tf.int32)

                        # Compute target Q-values
                        next_q_values = self.target_model(next_states)
                        max_next_q_values = tf.reduce_max(next_q_values, axis=1)
                        target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

                        with tf.GradientTape() as tape:
                            q_values = self.model(states)
                            action_masks = tf.one_hot(actions, action_size)
                            q_values = tf.reduce_sum(q_values * action_masks, axis=1)
                            loss = tf.keras.losses.MSE(target_q_values, q_values)

                        # Backpropagation
                        grads = tape.gradient(loss, self.model.trainable_variables)
                        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                    if done:
                        break

                # Update target network
                if episode % self.target_update == 0:
                    self.target_model.set_weights(self.model.get_weights())

                # Emit log update every 10 episodes
                if episode % 5 == 0:
                    log_message = (f"Episode {episode}, Total Reward: {total_reward:.2f}, Steps: {t}, "
                                   f"d_true: {self.env.d_true:.2f} mL, Stopping Position: {self.env.position:.2f} mL")
                    self.update_log.emit(log_message)

                # Emit animation update for specific runs (limit to first 10 episodes)
                if episode % 5 == 0:
                    frame_data = {
                        'positions': positions.copy(),
                        'd_true': self.env.d_true
                    }
                    self.update_animation.emit(frame_data)

            # Save the trained model
            self.model.save_weights('TitrationTraining2500.weights.h5')
            self.update_log.emit("Training completed and model saved as 'TitrationTraining2500.weights.h5'.")
        except Exception as e:
            self.update_log.emit(f"An error occurred during training: {e}")

# ----------------------------
# Evaluation Thread Definition
# ----------------------------

class EvaluationThread(QThread):
    update_log = pyqtSignal(str)
    update_animation = pyqtSignal(list, float)  # Positions and d_true
    update_results = pyqtSignal(str)

    def __init__(self, model, d_true, conc_naoh, volume_hcl):
        super().__init__()
        self.model = model
        self.conc_naoh = conc_naoh
        self.volume_hcl = volume_hcl
        self.env = RobotTrackEnv()
        self.d_true = d_true  # Correctly assign d_true
        self.env.set_equivalence_point(self.d_true)

    # Set user-defined d_true and calculate d_estimated

    def run(self):
        try:
            # Log the set equivalence point
            self.update_log.emit(
                f"Starting Evaluation with d_true = {self.env.d_true:.2f} mL and d_estimated = {self.env.d_estimated:.2f} mL")

            results = []
            final_errors = []

            for run in range(1, 4):
                self.env.set_equivalence_point(self.d_true)
                state = self.env.reset()
                positions = [self.env.position]
                total_reward = 0
                for t in range(1, 1501):  # Max steps per titration
                    action = self.select_action_greedy(state)
                    next_state, reward, done, _ = self.env.step(action)
                    state = next_state
                    total_reward += reward

                    # Record data
                    positions.append(self.env.position)

                    if done:
                        break

                final_error = abs(self.env.position - self.env.d_true)
                results.append({
                    'Run': run,
                    'Stopping Position': self.env.position,
                    'Final Error': final_error,
                    'Total Reward': total_reward,
                    'Steps': t
                })
                final_errors.append(final_error)
                self.update_log.emit(f"Titration Run {run}: Stopped at {self.env.position:.2f} mL, "
                                     f"Error: {final_error:.4f} mL, Reward: {total_reward:.2f}, Steps: {t}")

                # Emit animation update
                self.update_animation.emit(positions.copy(), self.env.d_true)

                time.sleep(2)

            # Check consistency
            if not self.is_consistent(final_errors):
                # Perform a fourth titration
                run = 4
                self.env.set_equivalence_point(self.d_true)
                state = self.env.reset()
                positions = [self.env.position]
                total_reward = 0
                for t in range(1, 1501):
                    action = self.select_action_greedy(state)
                    next_state, reward, done, _ = self.env.step(action)
                    state = next_state
                    total_reward += reward

                    # Record data
                    positions.append(self.env.position)

                    if done:
                        break

                final_error = abs(self.env.position - self.env.d_true)
                results.append({
                    'Run': run,
                    'Stopping Position': self.env.position,
                    'Final Error': final_error,
                    'Total Reward': total_reward,
                    'Steps': t
                })
                final_errors.append(final_error)
                self.update_log.emit(f"Titration Run {run}: Stopped at {self.env.position:.2f} mL, "
                                     f"Error: {final_error:.4f} mL, Reward: {total_reward:.2f}, Steps: {t}")

                # Emit animation update
                self.update_animation.emit(positions.copy(), self.env.d_true)

            # Calculate final concentration of HCl
            conc_hcl_final = (self.conc_naoh * self.env.d_true) / self.volume_hcl
            self.update_results.emit(f"Final Concentration of HCl: {conc_hcl_final:.4f} M")

        except Exception as e:
            self.update_log.emit(f"An error occurred during evaluation: {e}")

    def select_action_greedy(self, state):
        state = np.array([state], dtype=np.float32)
        q_values = self.model(state)
        return np.argmax(q_values[0])

    def is_consistent(self, errors, threshold=0.01):
        """
        Check if all final errors are within the threshold of the first error.
        """
        first_error = errors[0]
        return all(abs(e - first_error) < threshold for e in errors)


# ----------------------------
# Training Tab Definition
# ----------------------------

class TrainingTab(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.training_thread = None

    def init_ui(self):
        layout = QVBoxLayout()

        # Controls Layout
        controls_layout = QHBoxLayout()

        # Number of Episodes
        episodes_label = QLabel("Number of Episodes:")
        self.episodes_input = QSpinBox()
        self.episodes_input.setRange(1, 100000)
        self.episodes_input.setValue(200)  # Match console's 200 episodes

        # Start Training Button
        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self.start_training)

        # Save Model Button
        self.save_button = QPushButton("Save Model")
        self.save_button.clicked.connect(self.save_model)
        self.save_button.setEnabled(False)  # Disabled until training is done

        controls_layout.addWidget(episodes_label)
        controls_layout.addWidget(self.episodes_input)
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.save_button)

        # Animation Canvas
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)

        # Logging Area
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(150)

        layout.addLayout(controls_layout)
        layout.addWidget(QLabel("Training Animation:"))
        layout.addWidget(self.canvas)
        layout.addWidget(QLabel("Training Log:"))
        layout.addWidget(self.log)

        self.setLayout(layout)

    def start_training(self):
        num_episodes = self.episodes_input.value()
        self.log.append(f"Starting training for {num_episodes} episodes...")

        # Disable the start button to prevent multiple trainings
        self.start_button.setEnabled(False)
        self.save_button.setEnabled(False)

        # Initialize and start the training thread
        self.training_thread = TrainingThread(num_episodes)
        self.training_thread.update_log.connect(self.update_log)
        self.training_thread.update_animation.connect(self.update_animation)
        self.training_thread.start()

        # Re-enable the start button after training is done
        self.training_thread.finished.connect(self.on_training_finished)

    def save_model(self):
        if self.training_thread and self.training_thread.model:
            options = QFileDialog.Options()
            filepath, _ = QFileDialog.getSaveFileName(self, "Save Trained Model", "", "H5 Files (*.h5)", options=options)
            if filepath:
                try:
                    self.training_thread.model.save_weights(filepath)
                    self.log.append(f"Model saved to {filepath}")
                except Exception as e:
                    self.log.append(f"Error saving model: {e}")
                    QMessageBox.critical(self, "Save Model Error", f"An error occurred while saving the model:\n{e}")
        else:
            self.log.append("No trained model available to save.")
            QMessageBox.warning(self, "Save Model", "No trained model available to save.")

    def update_log(self, message):
        self.log.append(message)

    def update_animation(self, frame_data):
        try:
            # Clear figure
            self.figure.clear()

            # Ensure a subplot is added
            ax = self.figure.add_subplot(111)

            positions = frame_data['positions']
            d_true = frame_data['d_true']

            t = np.arange(len(positions)) * 0.1
            #x = np.linspace(0, d_true, len(positions))
            y = positions

            # Color mapping
            colors = np.exp(-200 * (np.array(y) - d_true - 0.05) ** 2)
            colors = np.clip(colors, 0.0, 1.0)
            cmap = plt.cm.Reds

            # Plot the scatter plot with the rectangular marker
            sc = ax.scatter(t, y, c=colors, cmap=cmap, s=200, marker='s', vmin=0.0, vmax=1.0)

            # Set axis limits and labels
            ax.set_xlim(0, 30)
            ax.set_ylim(0, d_true + 1)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Titrant Added (mL)')
            ax.set_title('Titration Animation')

            # Add colorbar
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('Indicator Color')

            # Draw the canvas
            self.canvas.draw()

        except Exception as e:
            print(f"Error in update_animation: {e}")

    def on_training_finished(self):
        self.log.append("Training thread has finished.")
        self.start_button.setEnabled(True)
        self.save_button.setEnabled(True)

# ----------------------------
# Evaluation Tab Definition
# ----------------------------

class EvaluationTab(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.evaluation_thread = None
        self.model = None  # Loaded model
        self.d_true = None
        self.d_est = None

    def init_ui(self):
        layout = QVBoxLayout()

        # Input Fields Layout
        inputs_layout = QHBoxLayout()

        # Concentration of HCl (Analyte)
        conc_hcl_label = QLabel("Concentration of HCl (M):")
        self.conc_hcl_input = QLineEdit()
        self.conc_hcl_input.setPlaceholderText("e.g., 0.1")

        # Volume of HCl
        volume_hcl_label = QLabel("Volume of HCl (mL):")
        self.volume_hcl_input = QLineEdit()
        self.volume_hcl_input.setPlaceholderText("e.g., 50")

        # Concentration of NaOH (Titrant)
        conc_naoh_label = QLabel("Concentration of NaOH (M):")
        self.conc_naoh_input = QLineEdit()
        self.conc_naoh_input.setPlaceholderText("e.g., 0.1")

        inputs_layout.addWidget(conc_hcl_label)
        inputs_layout.addWidget(self.conc_hcl_input)
        inputs_layout.addWidget(volume_hcl_label)
        inputs_layout.addWidget(self.volume_hcl_input)
        inputs_layout.addWidget(conc_naoh_label)
        inputs_layout.addWidget(self.conc_naoh_input)

        # Buttons Layout
        buttons_layout = QHBoxLayout()

        # Load Model Button
        self.load_model_button = QPushButton("Load Model")
        self.load_model_button.clicked.connect(self.load_model)

        # Sacrificial Run Button
        self.sacrificial_run_button = QPushButton("Sacrificial Run")
        self.sacrificial_run_button.clicked.connect(self.sacrificial_run)
        self.sacrificial_run_button.setEnabled(False)

        # Perform Titration Button
        self.perform_titration_button = QPushButton("Perform Titration")
        self.perform_titration_button.clicked.connect(self.perform_titration)
        self.perform_titration_button.setEnabled(False)

        buttons_layout.addWidget(self.load_model_button)
        buttons_layout.addWidget(self.sacrificial_run_button)
        buttons_layout.addWidget(self.perform_titration_button)

        # Animation Canvas
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)

        # Results Display
        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)
        self.results_display.setFixedHeight(150)

        # Logging Area
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(100)

        layout.addLayout(inputs_layout)
        layout.addLayout(buttons_layout)
        layout.addWidget(QLabel("Titration Animation:"))
        layout.addWidget(self.canvas)
        layout.addWidget(QLabel("Results:"))
        layout.addWidget(self.results_display)
        layout.addWidget(QLabel("Evaluation Log:"))
        layout.addWidget(self.log)

        self.setLayout(layout)

    def load_model(self):
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Trained Model", "", "H5 Files (*.h5)", options=options)
        if filepath:
            try:
                # Initialize the model architecture
                state_size = 4  # [position, velocity, warning_signal, d_estimated]
                action_size = 3  # [-0.5, 0, +0.5]
                self.model = DQN(state_size, action_size)
                # Build the model by passing a dummy input
                dummy_input = tf.constant([[0.0, 0.0, 0.0, 0.0]], dtype=tf.float32)
                self.model(dummy_input)
                # Load the trained weights
                self.model.load_weights(filepath)
                self.log.append(f"Model loaded from {filepath}")
                self.sacrificial_run_button.setEnabled(True)
            except Exception as e:
                self.log.append(f"Error loading model: {e}")
                QMessageBox.critical(self, "Load Model Error", f"An error occurred while loading the model:\n{e}")

    def sacrificial_run(self):
        """
        Perform a sacrificial run to calculate d_true and d_est.
        """
        try:
            conc_hcl = float(self.conc_hcl_input.text())
            volume_hcl = float(self.volume_hcl_input.text())
            conc_naoh = float(self.conc_naoh_input.text())

            if conc_hcl <= 0 or volume_hcl <= 0 or conc_naoh <= 0:
                raise ValueError("All inputs must be positive numbers.")

            # Equivalence point calculation: n_a = n_b => conc_a * volume_a = conc_b * volume_b
            d_true = (conc_hcl * volume_hcl) / conc_naoh  # mL

            # d_est is a result of a sacrificial run with some noise
            d_est = d_true + np.random.uniform(-0.5, 0.5)  # mL

            self.d_true = d_true
            self.d_est = d_est

            self.log.append(f"Sacrificial Run: d_true = {d_true:.2f} mL, d_est = {d_est:.2f} mL")
            self.perform_titration_button.setEnabled(True)
        except ValueError as ve:
            self.log.append(f"Input Error: {ve}")
            QMessageBox.warning(self, "Input Error", f"Please enter valid numeric values.\n{ve}")
        except Exception as e:
            self.log.append(f"Unexpected Error: {e}")
            QMessageBox.critical(self, "Error", f"An unexpected error occurred:\n{e}")

    def perform_titration(self):
        """
        Perform titration simulations and display results.
        """
        if not self.model or not hasattr(self, 'd_true') or not hasattr(self, 'd_est'):
            QMessageBox.warning(self, "Incomplete Setup", "Please load a model and perform a sacrificial run first.")
            return

        # Disable the perform titration button to prevent multiple runs
        self.perform_titration_button.setEnabled(False)

        try:
            conc_naoh = float(self.conc_naoh_input.text())
            volume_hcl = float(self.volume_hcl_input.text())
        except ValueError:
            self.log.append("Invalid concentration or volume inputs.")
            QMessageBox.warning(self, "Input Error", "Please enter valid numeric values for concentrations and volume.")
            self.perform_titration_button.setEnabled(True)
            return

        # Initialize and start the evaluation thread with user-defined d_true
        self.evaluation_thread = EvaluationThread(self.model, self.d_true, conc_naoh, volume_hcl)
        self.evaluation_thread.update_log.connect(self.update_log)
        self.evaluation_thread.update_animation.connect(self.update_animation)
        self.evaluation_thread.update_results.connect(self.update_results)
        self.evaluation_thread.start()

        # Re-enable the perform titration button after evaluation is done
        self.evaluation_thread.finished.connect(self.on_evaluation_finished)

    def update_log(self, message):
        self.log.append(message)

    import matplotlib.patches as patches

    def update_animation(self, positions, d_true):
        # Plot the titration animation
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        t = np.arange(len(positions)) * 0.1
        #x = np.linspace(0, d_true, len(positions))
        y = positions

        # Color mapping using "Reds" colormap based on concentration intensity
        colors = np.exp(-200 * (np.array(y) - d_true - 0.05)**2)
        colors = np.clip(colors, 0.0, 1.0)
        cmap = plt.cm.Reds

        sc = ax.scatter(t, y, c=colors, cmap=cmap,marker='s', s=200, vmin=0.0, vmax=1.0)
        ax.set_xlim(0, 30)
        ax.set_ylim(0, d_true + 1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Titrant Added (mL)')
        ax.set_title('Titration Animation')

        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Indicator Color')

        self.canvas.draw()

    def update_results(self, result_text):
        self.results_display.append(result_text)
        self.log.append("Titration completed.")
        self.perform_titration_button.setEnabled(True)

    def on_evaluation_finished(self):
        self.log.append("Evaluation thread has finished.")

# ----------------------------
# Main Window Definition
# ----------------------------

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Titrating Robot - Reinforcement Learning Lab")
        self.setGeometry(100, 100, 1200, 800)  # Width x Height

        self.layout = QVBoxLayout()
        self.tabs = QTabWidget()

        # Initialize tabs
        self.training_tab = TrainingTab()
        self.evaluation_tab = EvaluationTab()

        # Add tabs to the widget
        self.tabs.addTab(self.training_tab, "Training")
        self.tabs.addTab(self.evaluation_tab, "Evaluation")

        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

# ----------------------------
# Evaluation Thread Definition
# ----------------------------

# (Same as previously defined)

# ----------------------------
# Main Execution
# ----------------------------

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
