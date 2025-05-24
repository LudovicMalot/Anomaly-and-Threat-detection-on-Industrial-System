import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler

# ---------------------------
# 1. Data Loading & Preprocessing
# ---------------------------
def load_and_preprocess(filepath):
    """
    Reads CSV data, drops unwanted columns, scales features,
    and returns (X_scaled, y). Data is cast to float32.
    """
    df = pd.read_csv(filepath, sep=';', decimal=',')
    
    # Adjust these columns as needed based on your CSV structure
    X = df.drop(columns=[df.columns[0], 'Timestamp', 'Normal/Attack', 'anomaly_score'])
    y = df['Normal/Attack'].astype(int).values  # Convert labels to int
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    return X_scaled, y

# Load the dataset
X_scaled, y_true = load_and_preprocess("anomalies.csv")

# ---------------------------
# 2. Load the Trained Model
# ---------------------------
from stable_baselines3 import DQN

# Load the model produced from training (ensure this file exists)
model = DQN.load("anomaly_detection_dqn")

# ---------------------------
# 3. Make Predictions on the Whole Dataset
# ---------------------------
# The model's predict method accepts a NumPy array of observations.
predictions, _ = model.predict(X_scaled, deterministic=True)

# ---------------------------
# 4. Compute Evaluation Metrics
# ---------------------------
# Confusion Matrix
cm = confusion_matrix(y_true, predictions)
acc = accuracy_score(y_true, predictions)
recall = recall_score(y_true, predictions)      # Sensitivity, True Positive Rate
precision = precision_score(y_true, predictions)
f1 = f1_score(y_true, predictions)

print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_true, predictions))
print(f"Accuracy: {acc:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")

# ---------------------------
# 5. Plot and Save the Confusion Matrix
# ---------------------------
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
