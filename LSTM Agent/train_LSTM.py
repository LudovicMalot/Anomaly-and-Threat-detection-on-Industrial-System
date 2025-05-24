import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LSTM, GlobalAveragePooling1D, multiply
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tqdm.keras import TqdmCallback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight

# =============================================================================
# 1. Data Loading and Preprocessing
# =============================================================================
def load_and_preprocess_data(csv_path):
    """
    Loads a semicolon-separated CSV file, converts commas to dots in numeric fields,
    drops the timestamp column, maps labels to 0 (Normal) and 1 (Anomalie),
    and normalizes the sensor features.
    """
    df = pd.read_csv(csv_path, sep=';')
    
    # Assume the first column is timestamp and the last column is the label.
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
    
    # Drop the timestamp column and rows with missing values.
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
    Creates overlapping sequences from the data.
    Each sequence is of length `window_size`, and the label for the sequence
    is taken as the label at the final timestep.
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size + 1):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size-1])
    return np.array(X_seq), np.array(y_seq)

# =============================================================================
# 2. Build the Hybrid LSTM + Feature-Wise Attention Model
# =============================================================================
def build_model(window_size, num_features, use_focal_loss=False):
    """
    Builds a model that:
      - Processes sequential data with an LSTM layer
      - Pools over time with GlobalAveragePooling1D
      - Applies a squeeze-and-excitation block for feature-wise attention
      - Uses diminishing dense layers for further feature combination
      - Outputs a binary classification via a sigmoid activation
    """
    inputs = Input(shape=(window_size, num_features), name='input_layer')
    
    # LSTM layer to capture temporal dependencies.
    x = LSTM(64, return_sequences=True, name='lstm_layer')(inputs)
    
    # Global average pooling over time.
    x = GlobalAveragePooling1D(name='global_avg_pool')(x)
    
    # Squeeze-and-Excitation block (Feature-wise Attention)
    filters = 64  # same as LSTM units
    se = Dense(filters // 4, activation='relu', name='se_dense1')(x)
    se = Dense(filters, activation='sigmoid', name='se_dense2')(se)
    x = multiply([x, se], name='feature_attention')
    
    # Diminishing dense layers.
    x = Dense(32, activation='relu', name='dense_32')(x)
    x = BatchNormalization(name='bn_32')(x)
    x = Dropout(0.3, name='dropout_32')(x)
    
    x = Dense(16, activation='relu', name='dense_16')(x)
    x = BatchNormalization(name='bn_16')(x)
    x = Dropout(0.3, name='dropout_16')(x)
    
    # Final output layer for binary classification.
    outputs = Dense(1, activation='sigmoid', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='Hybrid_LSTM_Attention_Model')
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    
   
    loss_fn = 'binary_crossentropy'
    
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    return model

# =============================================================================
# 3. Main Training Routine
# =============================================================================
def main():
    # Update the CSV path as needed.
    csv_path = 'swat_combined_dataset_augmented.csv'
    
    print("Loading and preprocessing data...")
    X, y, scaler = load_and_preprocess_data(csv_path)
    print(f"Total samples: {X.shape[0]}, Features per sample: {X.shape[1]}")
    
    # Define the sequence window size (e.g., 10 timesteps).
    window_size = 10
    print(f"Creating sequences with window size = {window_size} ...")
    X_seq, y_seq = create_sequences(X, y, window_size)
    print(f"Sequences shape: {X_seq.shape}, Sequence labels shape: {y_seq.shape}")
    
    # Split into training and validation sets (80/20 split, stratified by label).
    X_train, X_val, y_train, y_val = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
    )
    print(f"Training sequences: {X_train.shape[0]}, Validation sequences: {X_val.shape[0]}")
    
    # Compute class weights to mitigate imbalance.
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: w for i, w in enumerate(weights)}
    print("Class weights:", class_weights)
    
    # Build the hybrid model.
    num_features = X_seq.shape[2]
    model = build_model(window_size, num_features, use_focal_loss=False)
    model.summary()
    
    # Callbacks: checkpoint, reduce learning rate, early stopping, and progress monitoring.
    checkpoint = ModelCheckpoint("hybrid_best_model.keras", monitor='val_accuracy', 
                                   save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1)
    tqdm_callback = TqdmCallback(verbose=1)
    
    # Train the model.
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=256,
        class_weight=class_weights,
        callbacks=[checkpoint, reduce_lr, early_stopping, tqdm_callback]
    )
    
    # Evaluate the model.
    print("Evaluating model on validation data...")
    loss, accuracy = model.evaluate(X_val, y_val, verbose=1)
    print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")
    
    # Generate predictions and compute evaluation metrics.
    y_pred_prob = model.predict(X_val)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    cm = confusion_matrix(y_val, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    report = classification_report(y_val, y_pred)
    print("Classification Report:")
    print(report)
    
    # Save the final model.
    model.save("final_hybrid_anomaly_model.keras")
    print("Final model saved as 'final_hybrid_anomaly_model.keras'.")

if __name__ == "__main__":
    main()
