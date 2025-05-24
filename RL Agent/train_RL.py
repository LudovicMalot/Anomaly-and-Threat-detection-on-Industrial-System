import gym
from gym import spaces
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

# For RL
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# ---------------------------
# 1. Data Loading & Preprocessing
# ---------------------------
def load_and_preprocess(filepath):
    """
    Reads CSV data, drops unwanted columns, scales features, returns (X_scaled, y).
    Ensures X is float32 for Mac MPS compatibility.
    """
    df = pd.read_csv(filepath, sep=';', decimal=',')
    
    # Adjust these columns as needed based on your CSV structure
    X = df.drop(columns=[df.columns[0], 'Timestamp', 'Normal/Attack', 'anomaly_score'])
    y = df['Normal/Attack'].astype(int).values  # Convert labels to int
    
    # Scale features and cast to float32
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    return X_scaled, y

# ---------------------------
# 2. Custom Gym Environment
# ---------------------------
class AnomalyDetectionEnv(gym.Env):
    """
    Environment simulating a human-in-the-loop anomaly detection:
      - Observations: sensor readings (float32).
      - Actions: 0 (predict Normal), 1 (predict Attack).
      - Rewards: 
          - Correct Attack: +2
          - Correct Normal: +1
          - False Alarm (predict Attack but it's Normal): -1
          - Missed Attack (predict Normal but it's Attack): -20 (heavily penalized)
    Optionally shuffles data at each reset so the agent doesn't just memorize order.
    """
    def __init__(self, data, labels, shuffle_on_reset=True):
        super(AnomalyDetectionEnv, self).__init__()
        self.data = data
        self.labels = labels
        self.n_samples = data.shape[0]
        self.current_index = 0
        self.shuffle_on_reset = shuffle_on_reset
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(2)  # 0 or 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(data.shape[1],),
            dtype=np.float32
        )
        
        # Reward parameters
        self.reward_if_correct_attack = 2.0
        self.reward_if_correct_normal = 1.0
        self.penalty_if_false_alarm   = -1.0
        self.penalty_if_missed_attack = -20.0

        # To hold indices if we shuffle each episode
        self.indices = np.arange(self.n_samples)

    def reset(self):
        # Shuffle data at each reset if desired
        if self.shuffle_on_reset:
            np.random.shuffle(self.indices)
        self.current_index = 0
        
        # Return first observation
        return self.data[self.indices[self.current_index]]

    def step(self, action):
        # Get true label for the current sample
        idx = self.indices[self.current_index]
        true_label = self.labels[idx]
        
        # Apply the reward logic
        if action == 1:  # Predict Attack
            if true_label == 1:
                reward = self.reward_if_correct_attack
            else:
                reward = self.penalty_if_false_alarm
        else:  # Predict Normal
            if true_label == 0:
                reward = self.reward_if_correct_normal
            else:
                reward = self.penalty_if_missed_attack
        
        self.current_index += 1
        done = (self.current_index >= self.n_samples)
        
        if not done:
            next_obs = self.data[self.indices[self.current_index]]
        else:
            # Return a zero-vector if the episode is done
            next_obs = np.zeros(self.data.shape[1], dtype=np.float32)
        
        return next_obs, reward, done, {}

# ---------------------------
# 3. Determine Device (MPS, CUDA, or CPU)
# ---------------------------
if torch.backends.mps.is_available():
    device = "mps"
    print("Using device: mps")
elif torch.cuda.is_available():
    device = "cuda"
    print("Using device: cuda")
else:
    device = "cpu"
    print("Using device: cpu")

# ---------------------------
# 4. Callbacks for Accuracy & TQDM
# ---------------------------
class AccuracyCallback(BaseCallback):
    """
    Evaluate the agent's "accuracy" on a separate dataset every eval_freq steps.
    This version does a single vectorized call to model.predict() for speed.
    Also records the initial accuracy at timestep 0.
    """
    def __init__(self, eval_data, eval_labels, eval_freq=1000, verbose=1):
        super(AccuracyCallback, self).__init__(verbose)
        self.eval_data = eval_data
        self.eval_labels = eval_labels
        self.eval_freq = eval_freq
        self.accuracies = []
        self.timesteps = []

    def _on_training_start(self) -> None:
        # Record initial accuracy at timestep 0
        acc = self.evaluate_accuracy()
        self.accuracies.append(acc)
        self.timesteps.append(0)
        if self.verbose > 0:
            print(f"\nStep 0: Accuracy = {acc:.4f}")

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            acc = self.evaluate_accuracy()
            self.accuracies.append(acc)
            self.timesteps.append(self.num_timesteps)
            if self.verbose > 0:
                print(f"\nStep {self.num_timesteps}: Accuracy = {acc:.4f}")
        return True

    def evaluate_accuracy(self):
        predictions, _ = self.model.predict(self.eval_data, deterministic=True)
        return np.mean(predictions == self.eval_labels)

class TqdmCallback(BaseCallback):
    """
    Use tqdm progress bar during training.
    """
    def __init__(self, total_timesteps, verbose=0):
        super(TqdmCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.tqdm_bar = tqdm(total=total_timesteps)

    def _on_step(self) -> bool:
        self.tqdm_bar.update(1)
        return True

    def _on_training_end(self) -> None:
        self.tqdm_bar.close()

# ---------------------------
# 5. Main Training Script
# ---------------------------
if __name__ == "__main__":
    # Load data
    X_scaled, y = load_and_preprocess("anomalies.csv")

    # Create the environment
    env = AnomalyDetectionEnv(X_scaled, y, shuffle_on_reset=True)

    # For evaluation, we can use the same data or (preferably) a held-out set
    eval_data = X_scaled
    eval_labels = y

    # Vectorize the environment for Stable-Baselines3
    # (DQN expects a vectorized environment)
    vec_env = DummyVecEnv([lambda: env])

    # Create the DQN model
    total_timesteps = 100_000  # Example: 100k steps (adjust as needed)
    model = DQN("MlpPolicy", vec_env, verbose=1, device=device)

    # Create callbacks
    # Now we evaluate accuracy every 1,000 steps.
    accuracy_callback = AccuracyCallback(eval_data, eval_labels, eval_freq=1000, verbose=1)
    tqdm_callback = TqdmCallback(total_timesteps=total_timesteps)

    # Train
    model.learn(total_timesteps=total_timesteps, callback=[accuracy_callback, tqdm_callback])

    # Save the trained model
    model.save("anomaly_detection_dqn")

    # Plot Accuracy Over Time
    plt.figure(figsize=(10, 6))
    plt.plot(accuracy_callback.timesteps, accuracy_callback.accuracies, marker='o', color='blue')
    plt.xlabel("Timesteps")
    plt.ylabel("Accuracy")
    plt.title("Agent Accuracy Over Time")
    plt.grid(True)
    plt.savefig("accuracy_over_time.png")

    # Evaluate the Trained Agent on One Full Episode
    obs = env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_reward += reward

    print("Total reward for the final evaluation episode:", total_reward)
