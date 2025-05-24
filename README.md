# 🔍 Industrial Anomaly Detection using LSTM & Reinforcement Learning (SWaT Dataset)

This project demonstrates a **hybrid approach** to anomaly detection in industrial control systems, combining **deep learning** (LSTM + attention) and **reinforcement learning** (DQN). The dataset used is the **Secure Water Treatment (SWaT)** dataset, a popular benchmark for cyber-physical anomaly detection.

---

## 🧠 What This Project Does

It detects anomalies in real-time sensor data using a **two-stage pipeline**:

1. **Hybrid Deep Learning Model**: An LSTM-based network with feature-wise attention first analyzes multivariate time series data to detect potential anomalies.
2. **Reinforcement Learning Agent**: The outputs of the LSTM model are then passed to a Deep Q-Network (DQN), which acts as a human-in-the-loop anomaly detector with tailored reward policies, refining the final decision.

This chaining (LSTM → RL) leverages the strengths of both deep learning for temporal feature extraction and reinforcement learning for adaptive decision-making.


## ⚙️ Technologies & Libraries

* Python, NumPy, Pandas, Matplotlib
* TensorFlow / Keras
* Scikit-learn
* Stable-Baselines3 (DQN)
* OpenAI Gym
* tqdm (for live progress monitoring)

---

## 📊 Hybrid LSTM-Attention Model

### Architecture Highlights:

* `LSTM (64 units)` for temporal pattern learning
* `GlobalAveragePooling` over timesteps
* `Squeeze-and-Excitation block` for feature-wise attention
* `Diminishing Dense layers + Dropout + BatchNorm`
* Final layer: `Sigmoid` for binary classification

### Training Strategy:

* **Input**: Sequences of 10 timesteps
* **Class Imbalance**: Handled using `class_weight` from scikit-learn
* **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, tqdm progress
* **Loss**: Binary Crossentropy
* **Evaluation**: Accuracy, Confusion Matrix, Classification Report

---

## 🤖 Reinforcement Learning Agent

### Custom Gym Environment

* Actions: `0 = Normal`, `1 = Attack`
* Rewards:

  * ✅ Correct Normal: `+1`
  * ✅ Correct Attack: `+2`
  * ❌ False Positive: `-1`
  * ❌ Missed Attack: `-20` (heavily penalized)

### Agent

* Algorithm: `DQN (Deep Q-Network)` from Stable-Baselines3
* Environment: Vectorized with `DummyVecEnv`
* Evaluation: Custom `AccuracyCallback` every 1000 steps

## 📈 Result

### Classification Report (LSTM Model)

![Classification Report LSTM](./LSTM%20Agent//confusion_matrix.png)

### Classification Report (RL Agent)

![Classification Report RL](./RL%20Agent//confusion_matrix.png)

### Reward Trends (RL Agent)

![Reward Trends](./RL%20Agent//accuracy_over_time.png)

---

## 🧠 Skills Demonstrated

* Time Series Preprocessing & Normalization
* Deep Learning with Attention Mechanisms
* Reinforcement Learning using OpenAI Gym + Stable Baselines
* Evaluation Metrics (F1, Confusion Matrix)
* Model Saving, Callbacks, Training Monitoring

---

## 📚 Dataset Source

* **[SWaT Dataset (Secure Water Treatment)](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)**

  * Simulated ICS/SCADA data from a real water treatment facility
  * Includes cyber-attacks and normal operations

---

## 🙋‍♂️ Author

👋 Hi! I'm a Ludovic Malot, a French engineer focused on AI/ML and cybersecurity applications. This project was a hands-on experiment to blend classic deep learning with reinforcement learning in a real-world industrial setting.

Feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/ludovic-malot/) or drop a ⭐ if this repo helped you!
