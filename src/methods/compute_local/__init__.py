from sklearn.linear_model import LinearRegression
import numpy as np
import tensorflow as tf

def compute_llm_classification(model, x, num_samples=100, perturb_std=0.1, kernel_width=0.1):
    x = np.array(x)
    n_features = len(x)
    x_tensor = tf.convert_to_tensor(x[None, :], dtype=tf.float32)

    X_perturbed = x + np.random.normal(0, perturb_std, size=(num_samples, n_features))  # Shape: (num_samples, n_features)
    X_perturbed_tensor = tf.convert_to_tensor(X_perturbed, dtype=tf.float32)

    base_pred = model(x_tensor)[0].numpy()
    predicted_class = np.argmax(base_pred)

    preds = model(X_perturbed_tensor)
    y_vals = preds[:, predicted_class].numpy()

    distances = tf.sqrt(tf.reduce_sum(tf.square(X_perturbed_tensor - x_tensor), axis=1))
    weights = np.sqrt(np.exp(-(distances ** 2) / (kernel_width ** 2)))

    reg = LinearRegression()
    reg.fit(X_perturbed, y_vals, sample_weight=weights)
    
    return reg.coef_

def compute_llm_regression(model, x, num_samples=100, perturb_std=0.1, kernel_width=0.1):
    x = np.array(x)
    n_features = len(x)
    x_tensor = tf.convert_to_tensor(x[None, :], dtype=tf.float32)

    X_perturbed = x + np.random.normal(0, perturb_std, size=(num_samples, n_features))
    X_perturbed_tensor = tf.convert_to_tensor(X_perturbed, dtype=tf.float32)

    base_pred = model(x_tensor)
    y_base = base_pred[0].numpy()

    preds = model(X_perturbed_tensor)
    y_vals = preds.numpy()

    distances = tf.sqrt(tf.reduce_sum(tf.square(X_perturbed_tensor - x_tensor), axis=1))
    weights = np.sqrt(np.exp(-(distances ** 2) / (kernel_width ** 2)))
    weights = np.where(weights < 1e-10, 1e-10, weights)

    if np.sum(weights) == 0:
        raise ValueError("Sum of weights is zero; adjust perturb_std or kernel_width.")
    
    reg = LinearRegression()
    reg.fit(X_perturbed, y_vals, sample_weight=weights)
    
    return reg.coef_.squeeze()

def compute_c_hp_classification(model, image, layer_name='hidden'):
    image = np.array(image)
    inputs = tf.convert_to_tensor(image[None, :], dtype=tf.float32)

    grad_model = tf.keras.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    # Compute gradients
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        conv_outputs, predictions = grad_model(inputs)
        classIdx = int(tf.argmax(predictions[0]).numpy())
        loss = predictions[:, classIdx]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise ValueError("Gradient is None")

    # Validate gradient values
    if tf.reduce_any(tf.math.is_nan(grads)) or tf.reduce_any(tf.math.is_inf(grads)):
        raise ValueError("Gradient contains NaN or Inf")

    c_hp = grads * tf.cast(conv_outputs, grads.dtype)
    c_hp_flat = tf.reshape(c_hp, [-1]).numpy()

    return c_hp_flat

def compute_c_hp_regression(model, image, layer_name='hidden'):
    image = np.array(image)
    inputs = tf.convert_to_tensor(image[None, :], dtype=tf.float32)

    grad_model = tf.keras.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        tape.watch(inputs)
        conv_outputs, predictions = grad_model(inputs)
        loss = tf.reduce_mean(predictions)

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise ValueError("Gradient is None; check the model or extraction layer.")

    if tf.reduce_any(tf.math.is_nan(grads)) or tf.reduce_any(tf.math.is_inf(grads)):
        raise ValueError("Gradient contains NaN or Inf; check the data or model.")

    c_hp = grads * tf.cast(conv_outputs, grads.dtype)
    c_hp_flat = tf.reshape(c_hp, [-1]).numpy()

    return c_hp_flat

__all__ = [
    'compute_llm_classification',
    'compute_llm_regression',
    'compute_c_hp_classification',
    'compute_c_hp_regression'
]