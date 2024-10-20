import numpy as np
import gym
from gym import spaces
import random
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt


# Custom Environment
class RobotTrackEnv(gym.Env):
    def __init__(self):
        super(RobotTrackEnv, self).__init__()
        # Dynamics constraints
        self.max_velocity = 1.25  # m/s
        self.min_velocity = 0.0  # m/s
        self.max_acceleration = 0.5  # m/s^2
        self.min_acceleration = -0.5  # m/s^2
        self.dt = 0.1  # time step in seconds
        self.action_space = spaces.Discrete(3)  # -0.5, 0, +0.5 m/s^2
        # Observation space: position, velocity, warning signal, d_estimated
        low_obs = np.array([0.0, self.min_velocity, 0.0, 9.5], dtype=np.float32)
        high_obs = np.array([np.inf, self.max_velocity, 1.0, 24.5], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
        self.reset()

    def reset(self):
        # Randomize the true value of d between 10.0 and 24.0 meters
        self.d_true = np.random.uniform(10.0, 24.0)
        # Recalculate d_estimated with updated d_true
        self.d_estimated = self.d_true + np.random.uniform(-0.5, 0.5)  # Adds noise
        self.position = 0.0  # Start at position 0
        self.velocity = 0.0  # Start at rest
        self.done = False
        self.total_time = 0.0
        self.warning = 0.0
        self.steps = 0
        return self._get_state()

    def step(self, action):
        # Map action to acceleration
        acceleration_values = [-0.5, 0.0, 0.5]  # m/s^2
        acceleration = acceleration_values[action]

        # Update velocity
        self.velocity += acceleration * self.dt
        self.velocity = np.clip(self.velocity, self.min_velocity, self.max_velocity)

        # Update position
        self.position += self.velocity * self.dt

        # Update total time
        self.total_time += self.dt

        # Calculate warning signal
        self.warning = self._calculate_warning()

        # Initialize reward
        reward = self.velocity * self.dt - 0.1  # Per-step reward

        # Check for early stopping
        if self.velocity == 0.0 and self.warning == 0.0 and self.position < self.d_true - 0.5:
            self.done = True
            reward += -50.0  # Severe penalty for stopping early
            return self._get_state(), reward, self.done, {}

        # Check for success
        if self.position >= self.d_true - 0.1 and self.position < self.d_true:
            self.done = True
            # Adjusted final reward without d_true dependency
            reward += 100.0 / self.total_time  # Encourages faster completion
        # Check if fallen off the cliff
        elif self.position > self.d_true:
            self.done = True
            reward += -100.0  # Severe penalty for failure

        # Increment steps
        self.steps += 1
        state = self._get_state()
        return state, reward, self.done, {}

    def _calculate_warning(self):
        x = self.position
        d_true = self.d_true  # Use the estimated value of d
        if x >= d_true - 0.40:
            w = 1 - np.exp(-20 * (x - d_true + 0.4) ** 2)
            return np.clip(w, 0.0, 1.0)
        else:
            return 0.0

    def _get_state(self):
        # Include d_estimated in the state, not d_true
        return np.array([self.position, self.velocity, self.warning, self.d_estimated], dtype=np.float32)

    def render(self, mode='human'):
        print(
            f"Position: {self.position:.2f} m, Velocity: {self.velocity:.2f} m/s, Warning: {self.warning:.2f}, d_estimated: {self.d_estimated:.2f} m")


# Neural Network for DQN (unchanged)
class DQN(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', input_shape=(state_size,))
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(action_size)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)


# Replay Memory (unchanged)
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


# Hyperparameters (adjusted if necessary)
num_episodes = 2500  # Increased number of episodes for better learning
max_steps = 1500  # Max steps per episode
batch_size = 64
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 1500  # Slower epsilon decay for more exploration
learning_rate = 0.001
target_update = 10
memory_capacity = 10000

# Initialize environment and agent
env = RobotTrackEnv()
state_size = env.observation_space.shape[0]  # Now includes d_estimated
action_size = env.action_space.n

policy_net = DQN(state_size, action_size)
target_net = DQN(state_size, action_size)
target_net.set_weights(policy_net.get_weights())

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
memory = ReplayMemory(memory_capacity)

steps_done = 0


def select_action(state):
    global steps_done
    epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
              np.exp(-1. * steps_done / epsilon_decay)
    steps_done += 1
    if random.random() < epsilon:
        return random.randrange(action_size)
    else:
        state = np.array([state], dtype=np.float32)
        q_values = policy_net(state)
        return np.argmax(q_values[0])


# Training Loop
episode_durations = []
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    for t in range(max_steps):
        action = select_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        # Store transition in memory
        memory.push(state, action, reward, next_state, done)
        state = next_state
        # Perform optimization
        if len(memory) >= batch_size:
            # Sample a batch of transitions from memory
            states, actions, rewards, next_states, dones = memory.sample(batch_size)
            states = np.array(states, dtype=np.float32)
            next_states = np.array(next_states, dtype=np.float32)
            rewards = rewards.astype(np.float32)
            actions = actions.astype(np.int32)
            dones = dones.astype(np.float32)

            # Compute target Q-values
            next_q_values = target_net(next_states)
            max_next_q_values = np.max(next_q_values, axis=1)
            target_q_values = rewards + (gamma * max_next_q_values * (1 - dones))

            with tf.GradientTape() as tape:
                q_values = policy_net(states)
                action_masks = tf.one_hot(actions, action_size)
                q_values = tf.reduce_sum(q_values * action_masks, axis=1)
                loss = tf.keras.losses.MSE(target_q_values, q_values)
            # Backpropagation
            grads = tape.gradient(loss, policy_net.trainable_variables)
            optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))
        if done:
            episode_durations.append(t + 1)
            break
    # Update target network
    if episode % target_update == 0:
        target_net.set_weights(policy_net.get_weights())
    # Print progress every 10 episodes
    if episode % 10 == 0:
        print(
            f"Episode {episode}, Total Reward: {total_reward:.2f}, Steps: {t + 1}, d_true: {env.d_true:.2f} m, Stopping Position: {env.position:.2f} m")

# Save the trained model
policy_net.save_weights('TitrationTraining2500.weights.h5')
print("Trained model saved as 'TitrationTraining2500.weights.h5'")
