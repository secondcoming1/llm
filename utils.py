import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import optimizers, Sequential
#from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Dropout, Input
#from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
#from keras.models import Model, load_model
from keras import regularizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, auc, roc_curve
#import os
from tqdm import tqdm
import json


def get_network_activity_info():
    feature_description = {
        'user': 'Unique identifier for a user in the network',
        'logon_on_own_pc_normal': 'Logon event on the user’s assigned PC during normal hours (binary)',
        'logon_on_other_pc_normal': 'Logon event on a different PC during normal hours (binary)',
        'logon_on_own_pc_off_hour': 'Logon event on the user’s assigned PC during off-hours (binary)',
        'logon_on_other_pc_off_hour': 'Logon event on a different PC during off-hours (binary)',
        'logon_hour': 'Hour of the day the logon event occurred (0-23)',
        'day_of_a_week': 'Day of the week when the activity occurred (0=Monday, 6=Sunday)',
        'device_connects_on_own_pc': 'Number of device connections to the user’s assigned PC',
        'device_connects_on_other_pc': 'Number of device connections to a different PC',
        'device_connects_on_own_pc_off_hour': 'Device connections to user’s assigned PC during off-hours',
        'device_connects_on_other_pc_off_hour': 'Device connections to a different PC during off-hours',
        'documents_copy_own_pc': 'Number of documents copied on the user’s assigned PC',
        'documents_copy_other_pc': 'Number of documents copied on a different PC',
        'exe_files_copy_own_pc': 'Number of executable files copied on the user’s assigned PC',
        'exe_files_copy_other_pc': 'Number of executable files copied on a different PC',
        'documents_copy_own_pc_off_hour': 'Documents copied on the user’s assigned PC during off-hours',
        'documents_copy_other_pc_off_hour': 'Documents copied on a different PC during off-hours',
        'exe_files_copy_own_pc_off_hour': 'Executable files copied on the user’s assigned PC during off-hours',
        'exe_files_copy_other_pc_off_hour': 'Executable files copied on a different PC during off-hours',
        'neutral_sites': 'Number of visits to neutral (non-malicious, non-job-search) websites',
        'job_search': 'Number of job search website visits',
        'hacking_sites': 'Number of hacking-related website visits',
        'neutral_sites_off_hour': 'Number of visits to neutral websites during off-hours',
        'job_search_off_hour': 'Number of job search website visits during off-hours',
        'hacking_sites_off_hour': 'Number of hacking-related website visits during off-hours',
        'total_emails': 'Total number of emails sent',
        'int_to_int_mails': 'Number of internal-to-internal emails sent',
        'int_to_out_mails': 'Number of internal-to-external emails sent',
        'out_to_int_mails': 'Number of external-to-internal emails received',
        'out_to_out_mails': 'Number of external-to-external emails sent',
        'internal_recipients': 'Number of unique internal email recipients',
        'external_recipients': 'Number of unique external email recipients',
        'distinct_bcc': 'Number of distinct blind carbon copy (BCC) recipients',
        'mails_with_attachments': 'Number of emails sent with attachments',
        'after_hour_mails': 'Number of emails sent after normal working hours',
        'role': 'User’s role level within the organization',
        'business_unit': 'Business unit the user belongs to',
        'functional_unit': 'Functional unit within the business',
        'department': 'Department the user belongs to',
        'team': 'Team assignment of the user',
        'O': 'Openness personality trait score',
        'C': 'Conscientiousness personality trait score',
        'E': 'Extraversion personality trait score',
        'A': 'Agreeableness personality trait score',
        'N': 'Neuroticism personality trait score'
    }
    return feature_description


def prep_data(data_in):

    # Split the input into lines.
    lines = data_in.strip().splitlines()
    header = None
    consolidated_rows = []  # Will hold the consolidated data for each block.
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Detect a header line. Use it only once.
        if line.lower().startswith("user"):
            # Set the header only if not already set.
            if header is None:
                header = line.split()
            i += 1  # Move to the first data row.

            # Collect the next 3 data rows for this block.
            block_data = []
            for _ in range(3):
                if i >= len(lines):
                    break
                row_line = lines[i].strip()
                # If we encounter another header early, break.
                if row_line.lower().startswith("user"):
                    break
                values = row_line.split()
                # Convert values to float when possible.
                converted_values = []
                for val in values:
                    try:
                        converted_values.append(float(val))
                    except ValueError:
                        converted_values.append(val)
                block_data.append(converted_values)
                i += 1

            # Consolidate the block by columns.
            # For each column (by index), gather the values from each of the 3 rows.
            consolidated_block = []
            for col_idx in range(len(header)):
                col_values = [row[col_idx] for row in block_data]
                consolidated_block.append(col_values)
            # Append the consolidated block (as a list of column values) to our rows.
            consolidated_rows.append(consolidated_block)
        else:
            i += 1

    # Build the final structure without repeating column names in each row.
    final_data = {
        "header": header,
        "rows": consolidated_rows
    }

    # Display the final result.
    print("Header:")
    print(final_data["header"])
    print("\nConsolidated Rows:")
    for idx, row in enumerate(final_data["rows"], start=1):
        print(f"Row {idx}:")
        print(row)
    return final_data



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
    learning_rate=0.01, iterations=500, threshold=0.04, immutable_features=[], diversity_weight=0.01
):
    """
    Generates diverse counterfactuals while ensuring they are classified as normal 
    (i.e., reconstruction error is below the given threshold). A diversity term is added 
    to the loss to encourage counterfactuals to be different from one another.
    """
    counterfactuals = []
    print("\n🚀 Generating Counterfactuals...\n")

    # Convert the input sequence to a TensorFlow tensor.
    sequence_tf = tf.convert_to_tensor(sequence, dtype=tf.float32)
    
    # Determine indices for immutable features.
    immutable_indices = np.array(
        [feature_names.index(feature) for feature in immutable_features if feature in feature_names], dtype=int
    )
    original_immutable_values = sequence[:, :, immutable_indices].astype(np.float32)

    for cf_idx in tqdm(range(num_counterfactuals), desc="Generating Counterfactuals", unit="cf"):
        # Initialize the candidate counterfactual with the original sequence.
        seq_cf = tf.Variable(sequence_tf, dtype=tf.float32)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        for _ in tqdm(range(iterations), desc=f"Optimizing CF {cf_idx+1}/{num_counterfactuals}", unit="step", leave=False):
            with tf.GradientTape() as tape:
                tape.watch(seq_cf)
                reconstruction = model(seq_cf)
                reconstruction_loss = tf.reduce_mean(tf.abs(reconstruction - seq_cf))
                
                # Apply a penalty if reconstruction error is above the threshold.
                penalty = tf.maximum(reconstruction_loss - threshold, 0) * 10.0

                # Compute diversity loss: for each already finalized counterfactual,
                # penalize if the current candidate is too similar.
                diversity_loss = 0.0
                if len(counterfactuals) > 0:
                    for prev in counterfactuals:
                        prev_tensor = tf.convert_to_tensor(prev, dtype=tf.float32)
                        distance = tf.norm(seq_cf - prev_tensor)
                        diversity_loss += 1.0 / (distance + 1e-8)  # Avoid division by zero

                # Total loss includes reconstruction loss, anomaly penalty, and diversity term.
                total_loss = reconstruction_loss + penalty + diversity_weight * diversity_loss

            grads = tape.gradient(total_loss, seq_cf)
            if grads is not None:
                grads_numpy = grads.numpy()
                # Zero out gradients for immutable features so they remain unchanged.
                if len(immutable_indices) > 0:
                    grads_numpy[:, :, immutable_indices] = 0  
                grads_tf = tf.convert_to_tensor(grads_numpy, dtype=tf.float32)
                optimizer.apply_gradients([(grads_tf, seq_cf)])

                # Clip the candidate values to [0, 1].
                seq_cf.assign(tf.clip_by_value(seq_cf, 0, 1))

                # Restore immutable features using their original values.
                updates = tf.convert_to_tensor(original_immutable_values, dtype=tf.float32)
                indices = np.array([[b, t, f] 
                                     for b in range(seq_cf.shape[0]) 
                                     for t in range(seq_cf.shape[1]) 
                                     for f in immutable_indices])
                seq_cf.assign(tf.tensor_scatter_nd_update(seq_cf, indices, tf.reshape(updates, [-1])))

            # Early stopping if the candidate's reconstruction error is below the threshold.
            if reconstruction_loss.numpy() < threshold:
                print(f"✅ CF {cf_idx+1} is now normal (MSE={reconstruction_loss.numpy():.5f}) - stopping early")
                break

        counterfactuals.append(seq_cf.numpy())

    print("\n✅ Counterfactual Generation Complete!")

    # Reshape and inverse-transform to get the counterfactuals back on the original scale.
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




def format_data_for_llm(anomalous_sequence, counterfactuals, feature_names, scaler):
    # Convert anomalous sequence to original scale
    anomalous_original = scaler.inverse_transform(anomalous_sequence.reshape(-1, len(feature_names)))

    # Convert counterfactual DataFrames to NumPy arrays does not need scaling as it is already scaled correctly
    counterfactuals_original = [
        cf_df.to_numpy().astype(float).reshape(-1, len(feature_names))  # Convert to float64
        for cf_df in counterfactuals  # Convert DataFrame to NumPy array
    ]

    # Convert to list of dictionaries with rounded values (ensuring float conversion)
    anomalous_session = [
        {feature_names[i]: round(float(value), 2) for i, value in enumerate(row)}
        for row in anomalous_original
    ]

    counterfactuals_list = [
        [
            {feature_names[i]: round(float(value), 2) for i, value in enumerate(row)}
            for row in cf_seq
        ]
        for cf_seq in counterfactuals_original
    ]

    # Create dictionary for LLM input
    formatted_data = {
        "anomalous_session": anomalous_session,
        "counterfactuals": counterfactuals_list
    }

    # Convert to JSON string (ensure_ascii=False to support special characters)
    return json.dumps(formatted_data, indent=4, ensure_ascii=False)
