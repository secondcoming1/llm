def generate_counterfactual(model, sequence, scaler, feature_names, learning_rate=0.01, iterations=500, immutable_features=[]):
    sequence = tf.Variable(sequence, dtype=tf.float32)
    feature_indices = np.array([feature_names.index(f) for f in immutable_features if f in feature_names])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    for _ in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(sequence)
            reconstruction = model(sequence)
            loss = tf.reduce_mean(tf.abs(reconstruction - sequence))
        grads = tape.gradient(loss, sequence).numpy()
        if len(feature_indices) > 0:
            grads[:, :, feature_indices] = 0
        optimizer.apply_gradients([(tf.convert_to_tensor(grads), sequence)])
    counterfactual = scaler.inverse_transform(sequence.numpy()[:, -1, :])
    #counterfactual_full_sequence = scaler.inverse_transform(counterfactual.reshape(-1, counterfactual.shape[-1]))
    return pd.DataFrame(counterfactual, columns=feature_names)
    #return pd.DataFrame(counterfactual_full_sequence, columns=feature_names)