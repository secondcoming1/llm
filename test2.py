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

LABELS = ["Normal", "Anomaly"]

def read_data(path):
    df1 = pd.read_csv(path)
    df1 = df1.drop(['start_ts', 'session_duration'], axis=1)
    df1 = df1.fillna(0)
    df1['role'] = df1['role'].astype('category').cat.codes
    df1['user'] = df1['user'].astype('category').cat.codes
    return df1

def temporalize(in_X, lb):
    return np.array([in_X[i - lb:i, :] for i in range(lb, len(in_X) + 1)])

def get_train_test_data(train, test, lookback):
    sc = MinMaxScaler(feature_range=(0, 1))
    train_scaled = sc.fit_transform(train.drop(columns=['class', 'type'], errors='ignore'))
    test_scaled = sc.transform(test.drop(columns=['class', 'type'], errors='ignore'))

    X_train = np.array([train_scaled[i - lookback:i, :] for i in range(lookback, len(train_scaled))])
    X_test = np.array([test_scaled[i - lookback:i, :] for i in range(lookback, len(test_scaled))])
    y_test = test.iloc[lookback - 1:, test.columns.get_loc('class')].values

    return X_train, X_test, y_test, sc

def generate_counterfactual(model, sequence, scaler, feature_names, learning_rate=0.01, iterations=500, immutable_features=[]):
    """
    Generates a counterfactual sequence by modifying input features to reduce reconstruction error.

    Parameters:
    - model: Trained LSTM autoencoder.
    - sequence: Anomalous sequence (numpy array, shape=(1, timesteps, features)).
    - scaler: Scaler used for normalization (to inverse transform after modification).
    - feature_names: List of feature names corresponding to the sequence.
    - learning_rate: Step size for gradient updates.
    - iterations: Number of optimization steps.
    - immutable_features: List of feature names that should not change.

    Returns:
    - pd.DataFrame: Counterfactual sequence (all 3 time steps).
    """

    sequence = tf.Variable(sequence, dtype=tf.float32)

    feature_indices = np.array([feature_names.index(feature) for feature in immutable_features if feature in feature_names])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for i in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(sequence)
            reconstruction = model(sequence)
            loss = tf.reduce_mean(tf.abs(reconstruction - sequence))  

        grads = tape.gradient(loss, sequence)
        grads = grads.numpy()

        # Ensure immutable features do not change
        if len(feature_indices) > 0:
            grads[:, :, feature_indices] = 0  

        grads = tf.convert_to_tensor(grads)
        optimizer.apply_gradients([(grads, sequence)])

        # **Keep values within a realistic range**
        sequence.assign(tf.clip_by_value(sequence, 0, 1))  # Keep values between 0 and 1

    counterfactual = sequence.numpy()

    # **Explicitly restore immutable features**
    for feature in immutable_features:
        feature_idx = feature_names.index(feature)
        counterfactual[:, :, feature_idx] = sequence[:, :, feature_idx]  

    # ✅ Apply inverse transformation to all time steps
    counterfactual_full_sequence = scaler.inverse_transform(counterfactual.reshape(-1, counterfactual.shape[-1]))
    
    # ✅ Fix integer features (e.g., logon hour, day of week)
    integer_features = ["logon_hour", "day_of_a_week", "team"]
    counterfactual_full_sequence[:, [feature_names.index(f) for f in integer_features]] = \
        np.round(counterfactual_full_sequence[:, [feature_names.index(f) for f in integer_features]]).astype(int)

    return pd.DataFrame(counterfactual_full_sequence, columns=feature_names)

###############


def evaluate_model_on_sequences(model, anomalous_sequence, counterfactual_sequence, feature_names, scaler):
    """
    Evaluates the model on both the original anomaly and the counterfactual.
    Also computes and returns the sorted feature-wise differences between the anomaly and counterfactual.

    Args:
        model: Trained LSTM autoencoder.
        anomalous_sequence: The detected anomalous sequence (shape: (1, timesteps, features)).
        counterfactual_sequence: The generated counterfactual sequence (shape: (1, timesteps, features)).
        feature_names: List of feature names.
        scaler: The MinMaxScaler used for normalization (to inverse transform for meaningful feature comparison).

    Returns:
        A dictionary with:
        - 'anomaly_reconstruction_error': Mean reconstruction error of the anomaly.
        - 'counterfactual_reconstruction_error': Mean reconstruction error of the counterfactual.
        - 'feature_differences': DataFrame of sorted feature differences.
    """

    # Get model predictions (reconstructions)
    anomaly_reconstructed = model.predict(anomalous_sequence)
    counterfactual_reconstructed = model.predict(counterfactual_sequence)

    # Compute reconstruction errors (Mean Absolute Error)
    anomaly_error = np.mean(np.abs(anomalous_sequence - anomaly_reconstructed))
    counterfactual_error = np.mean(np.abs(counterfactual_sequence - counterfactual_reconstructed))

    # **Inverse transform** to get original scale values for meaningful difference analysis
    anomaly_original = scaler.inverse_transform(anomalous_sequence.reshape(-1, len(feature_names)))
    counterfactual_original = scaler.inverse_transform(counterfactual_sequence.reshape(-1, len(feature_names)))

    # Compute absolute differences between anomaly and counterfactual
    feature_differences = np.abs(anomaly_original - counterfactual_original)
    
    # Create a DataFrame for readability
    diff_df = pd.DataFrame(feature_differences, columns=feature_names)
    
    # Compute mean difference across timesteps (if lookback > 1)
    mean_differences = diff_df.mean().sort_values(ascending=False)  # Sort descending
    
    # Convert to DataFrame
    sorted_feature_differences = pd.DataFrame(mean_differences, columns=['Difference'])

    # Print summary
    print("\n🔍 **Model Evaluation Results:**")
    print(f"🛑 Anomalous Session Reconstruction Error: {anomaly_error:.6f}")
    print(f"✅ Counterfactual Session Reconstruction Error: {counterfactual_error:.6f}")

    if counterfactual_error < anomaly_error:
        print("✔️ Counterfactual successfully reduces the anomaly score! 🎯")
    else:
        print("⚠️ Counterfactual did not significantly reduce the anomaly score.")

    print("\n📊 **Top Feature Differences Between Anomaly and Counterfactual:**")
    print(sorted_feature_differences.to_string())

    return {
        "anomaly_reconstruction_error": anomaly_error,
        "counterfactual_reconstruction_error": counterfactual_error,
        "feature_differences": sorted_feature_differences
    }

#############

def main():
    path = "/home/sathish/UEBA/data/data.csv"
    df = read_data(path)
    train_data, test_data = df.iloc[:276388], df.iloc[276388:]
    lookback = 3
    X_train, X_test, y_test, scaler = get_train_test_data(train_data, test_data, lookback)
    n_features = X_train.shape[2]
    model_path = "lstm_autoencoder.h5"
    
    if os.path.exists(model_path):
        lstm_model = load_model(model_path, custom_objects={'MeanSquaredError': tf.keras.losses.MeanSquaredError()})
    else:
        inputs = Input(shape=(lookback, n_features))
        encoded = LSTM(16, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.00))(inputs)
        encoded = LSTM(8, activation='relu', return_sequences=False)(encoded)
        decoded = RepeatVector(lookback)(encoded)
        decoded = LSTM(8, activation='relu', return_sequences=True)(decoded)
        decoded = LSTM(16, activation='relu', return_sequences=True)(decoded)
        output = TimeDistributed(Dense(n_features))(decoded)
        lstm_model = Model(inputs, output)
        lstm_model.compile(optimizer=optimizers.Adam(0.0001), loss=tf.keras.losses.MeanSquaredError())
        lstm_model.fit(X_train, X_train, epochs=100, batch_size=256, validation_split=0.12, verbose=1)
        lstm_model.save(model_path)
    
    test_x_predictions = lstm_model.predict(X_test)
    mse = np.mean(np.power(X_test - test_x_predictions, 2), axis=(1, 2))
    threshold = 0.04
    pred_y = (mse > threshold).astype(int)
    anomaly_idx = np.where(pred_y == 1)[0][0]
    
    if anomaly_idx >= lookback:
        anomalous_sequence = X_test[anomaly_idx]  # Directly select the sequence
        
        # Ensure correct shape before reshaping
        if anomalous_sequence.shape != (lookback, X_train.shape[2]):  
            raise ValueError(f"Unexpected shape: {anomalous_sequence.shape}. Expected ({lookback}, {X_train.shape[2]})")
        
        anomalous_sequence = anomalous_sequence.reshape(1, lookback, X_train.shape[2])  # Final reshape
        print(f"Fixed anomalous_sequence shape: {anomalous_sequence.shape}")  # Should be (1, 3, 45)
    else:
        raise ValueError(f"Anomaly index {anomaly_idx} is too small for lookback={lookback}.")

    
    immutable_features = ["user","role","O","C","E","A","N"]
    feature_names = train_data.drop(columns=['class', 'type'], errors='ignore').columns.tolist()
    counterfactual_example = generate_counterfactual(lstm_model, anomalous_sequence, scaler, feature_names, immutable_features=immutable_features)

    print("Counterfactual Example:")
    print(counterfactual_example.to_string())



    # Reshape counterfactual for model evaluation
    counterfactual_array = counterfactual_example


    # **Normalize the counterfactual sequence** using the same scaler as training
    counterfactual_array = scaler.transform(counterfactual_array)
    # Reshape again to (1, timesteps, features) for model evaluation
    counterfactual_array = counterfactual_array.reshape(1, lookback, X_train.shape[2])
    # Evaluate the model on the original and counterfactual sequences
    evaluation_results = evaluate_model_on_sequences(
            lstm_model, anomalous_sequence, counterfactual_array, feature_names, scaler
    )

    print("\n📊 **Sorted Feature Differences:**")
    print(evaluation_results["feature_differences"])
if __name__ == "__main__":
    main()
