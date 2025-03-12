import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import optimizers, Sequential
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Dropout, Input
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.models import Model, load_model
from keras import regularizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, auc, roc_curve
import os
from tqdm import tqdm

LABELS = ["Normal", "Anomaly"]

### 🟢 Step 1: Data Preprocessing Functions ###
def read_data(path):
    """Reads and preprocesses dataset."""
    df1 = pd.read_csv(path)
    df1 = df1.drop(['start_ts', 'session_duration'], axis=1)
    df1 = df1.fillna(0)
    df1['role'] = df1['role'].astype('category').cat.codes
    df1['user'] = df1['user'].astype('category').cat.codes
    return df1

def get_train_test_data(train, test, lookback):
    """Prepares training and testing sequences."""
    sc = MinMaxScaler(feature_range=(0, 1))
    train_scaled = sc.fit_transform(train.drop(columns=['class', 'type'], errors='ignore'))
    test_scaled = sc.transform(test.drop(columns=['class', 'type'], errors='ignore'))

    # Creating time-series sequences
    X_train = np.array([train_scaled[i - lookback:i, :] for i in range(lookback, len(train_scaled))])
    X_test = np.array([test_scaled[i - lookback:i, :] for i in range(lookback, len(test_scaled))])
    y_test = test.iloc[lookback - 1:, test.columns.get_loc('class')].values

    return X_train, X_test, y_test, sc

### 🟢 Step 2: Counterfactual Generation Function ###

def generate_diverse_counterfactuals(
    model, sequence, scaler, feature_names, num_counterfactuals=5, 
    learning_rate=0.01, iterations=500, immutable_features=[], diversity_weight=0.01
):
    """
    Generates diverse counterfactuals while ensuring they are classified as normal (MSE < 0.04).
    """
    counterfactuals = []
    print("\n🚀 Generating Counterfactuals...\n")

    sequence_tf = tf.convert_to_tensor(sequence, dtype=tf.float32)
    immutable_indices = np.array(
        [feature_names.index(feature) for feature in immutable_features if feature in feature_names], dtype=int
    )
    original_immutable_values = sequence[:, :, immutable_indices].astype(np.float32)

    for cf_idx in tqdm(range(num_counterfactuals), desc="Generating Counterfactuals", unit="cf"):
        seq_cf = tf.Variable(sequence_tf, dtype=tf.float32)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        for _ in tqdm(range(iterations), desc=f"Optimizing CF {cf_idx+1}/{num_counterfactuals}", unit="step", leave=False):
            with tf.GradientTape() as tape:
                tape.watch(seq_cf)
                reconstruction = model(seq_cf)

                # **🔥 New Loss: Ensure Counterfactual Becomes Normal**
                reconstruction_loss = tf.reduce_mean(tf.abs(reconstruction - seq_cf))

                # **🔴 Penalty if reconstruction error > 0.04**
                penalty = tf.maximum(reconstruction_loss - 0.04, 0) * 10.0  # Large penalty for staying anomalous

                # 🔥 **Total Loss**
                total_loss = reconstruction_loss + penalty  # Added penalty

            grads = tape.gradient(total_loss, seq_cf)
            
            if grads is not None:
                grads_numpy = grads.numpy()
                if len(immutable_indices) > 0:
                    grads_numpy[:, :, immutable_indices] = 0  
                grads_tf = tf.convert_to_tensor(grads_numpy, dtype=tf.float32)
                optimizer.apply_gradients([(grads_tf, seq_cf)])

                # ✅ **Clip Values for Valid Ranges**
                seq_cf.assign(tf.clip_by_value(seq_cf, 0, 1))

                # ✅ **Restore Immutable Features**
                updates = tf.convert_to_tensor(original_immutable_values, dtype=tf.float32)
                indices = np.array([[b, t, f] for b in range(seq_cf.shape[0]) for t in range(seq_cf.shape[1]) for f in immutable_indices])
                seq_cf.assign(tf.tensor_scatter_nd_update(seq_cf, indices, tf.reshape(updates, [-1])))

            # **✅ Stop Early If Counterfactual is Normal**
            if reconstruction_loss.numpy() < 0.04:
                print(f"✅ CF {cf_idx+1} is now normal (MSE={reconstruction_loss.numpy():.5f}) - stopping early")
                break

        counterfactuals.append(seq_cf.numpy())

    print("\n✅ Counterfactual Generation Complete!")

    counterfactuals = np.array(counterfactuals)
    reshaped_cf = counterfactuals.reshape(-1, sequence.shape[-1])
    counterfactuals_original_scale = scaler.inverse_transform(reshaped_cf)
    counterfactuals_original_scale = counterfactuals_original_scale.reshape(num_counterfactuals, sequence.shape[1], sequence.shape[2])

    return [pd.DataFrame(cf, columns=feature_names) for cf in counterfactuals_original_scale]




### 🟢 Step 3: Model Evaluation Function ###
def evaluate_model_on_sequences(model, anomalous_sequence, counterfactual_sequences, feature_names, scaler):
    """
    Evaluates both the original anomaly and generated counterfactuals.
    """
    anomaly_reconstructed = model.predict(anomalous_sequence)
    anomaly_error = np.mean(np.abs(anomalous_sequence - anomaly_reconstructed))

    evaluation_results = []
    for cf_sequence in counterfactual_sequences:
        cf_sequence = scaler.transform(cf_sequence)
        cf_sequence = cf_sequence.reshape(1, anomalous_sequence.shape[1], anomalous_sequence.shape[2])
        cf_reconstructed = model.predict(cf_sequence)
        cf_error = np.mean(np.abs(cf_sequence - cf_reconstructed))

        anomaly_original = scaler.inverse_transform(anomalous_sequence.reshape(-1, len(feature_names)))
        counterfactual_original = scaler.inverse_transform(cf_sequence.reshape(-1, len(feature_names)))
        feature_differences = np.abs(anomaly_original - counterfactual_original)
        diff_df = pd.DataFrame(feature_differences, columns=feature_names)
        mean_differences = diff_df.mean().sort_values(ascending=False)
        sorted_feature_differences = pd.DataFrame(mean_differences, columns=['Difference'])

        evaluation_results.append({
            "counterfactual_reconstruction_error": cf_error,
            "feature_differences": sorted_feature_differences
        })

    print("\n🔍 **Model Evaluation Results:**")
    print(f"🛑 Anomalous Session Reconstruction Error: {anomaly_error:.6f}")

    for idx, result in enumerate(evaluation_results):
        print(f"\n✅ Counterfactual {idx+1} Reconstruction Error: {result['counterfactual_reconstruction_error']:.6f}")
        print(result["feature_differences"].to_string())
    return evaluation_results

### 🟢 Step 4: Main Execution ###
def main():
    path = "/home/sathish/UEBA/data/data.csv"
    df = read_data(path)
    train_data, test_data = df.iloc[:276388], df.iloc[276388:]
    lookback = 3
    X_train, X_test, y_test, scaler = get_train_test_data(train_data, test_data, lookback)
    n_features = X_train.shape[2]

    # Load or train model
    model_path = "lstm_autoencoder.h5"
    lstm_model = load_model(model_path, custom_objects={'MeanSquaredError': tf.keras.losses.MeanSquaredError()}) if os.path.exists(model_path) else None
    
    mse = np.mean(np.power(X_test - lstm_model.predict(X_test), 2), axis=(1, 2))
    threshold = 0.04
    anomaly_idx = np.where(mse > threshold)[0][0]
    anomalous_sequence = X_test[anomaly_idx].reshape(1, lookback, X_train.shape[2])

    feature_names = train_data.drop(columns=['class', 'type'], errors='ignore').columns.tolist()
    counterfactual_examples = generate_diverse_counterfactuals(lstm_model, anomalous_sequence, scaler, feature_names, immutable_features=["user", "role", "O", "C", "E", "A", "N"])

    evaluate_model_on_sequences(lstm_model, anomalous_sequence, counterfactual_examples, feature_names, scaler)

if __name__ == "__main__":
    main()
