import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, 
                             accuracy_score, recall_score, precision_score, f1_score)
from sklearn.preprocessing import StandardScaler

# ---------------------------
# 1. Data Loading and Preprocessing
# ---------------------------
def load_and_preprocess_data(csv_path):
    """
    Loads a semicolon-separated CSV file, converts commas to dots in numeric fields,
    drops the timestamp column, maps labels to 0 (Normal) and 1 (Anomalie),
    and normalizes the sensor features.
    """
    df = pd.read_csv(csv_path, sep=';')
    
    # Assume first column is timestamp and the last column is the label.
    sensor_columns = df.columns[1:-1]
    
    # Replace comma with dot and convert sensor columns to numeric.
    for col in sensor_columns:
        df[col] = df[col].astype(str).str.replace(',', '.')
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Process label column: map strings if necessary.
    label_col = df.columns[-1]
    try:
        df[label_col] = pd.to_numeric(df[label_col])
    except Exception:
        df[label_col] = df[label_col].map({'Normal': 0, 'Anomalie': 1})
    
    # Drop timestamp column and any rows with missing values.
    df = df.drop(columns=df.columns[0]).dropna()
    
    # Separate features and labels.
    X = df.iloc[:, :-1].values  # sensor features
    y = df.iloc[:, -1].values   # binary labels
    
    # Normalize features (zero mean, unit variance).
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def create_sequences(X, y, window_size):
    """
    Creates overlapping sequences of length `window_size` from the data.
    The label for each sequence is taken as the label at the final timestep.
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size + 1):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size-1])
    return np.array(X_seq), np.array(y_seq)

# ---------------------------
# 2. Testing Routine
# ---------------------------
def main():
    csv_path = 'swat_combined_dataset_augmented.csv'  # Update this path if needed.
    window_size = 10       # Must match the window size used during training.
    
    print("Loading and preprocessing data...")
    X, y, scaler = load_and_preprocess_data(csv_path)
    
    print("Creating sequences...")
    X_seq, y_seq = create_sequences(X, y, window_size)
    
    # Load the trained model.
    model = tf.keras.models.load_model("final_hybrid_anomaly_model.keras", compile=False)
    # Recompile if necessary.
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    print("Running predictions on the entire dataset...")
    y_pred_prob = model.predict(X_seq, batch_size=256)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Flatten to 1D arrays.
    y_true = y_seq.flatten()
    predictions = y_pred.flatten()
    
    # Compute evaluation metrics.
    cm = confusion_matrix(y_true, predictions)
    report = classification_report(y_true, predictions)
    acc = accuracy_score(y_true, predictions)
    recall = recall_score(y_true, predictions)
    precision = precision_score(y_true, predictions)
    f1 = f1_score(y_true, predictions)
    
    # Print the results.
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    print(f"Accuracy: {acc:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # ---------------------------
    # 3. Plot and Save the Confusion Matrix
    # ---------------------------
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")

if __name__ == "__main__":
    main()
