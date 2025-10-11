import tensorflow as tf
import numpy as np
from tqdm.auto import tqdm

def compute_sens(model, dataset, batch_size=128):
    n_samples, n_features = dataset.shape
    sens = np.zeros(n_features, dtype=np.float32)

    dataset_tensor = tf.convert_to_tensor(dataset, dtype=tf.float32)

    num_batches = (n_samples + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch = dataset_tensor[start_idx:end_idx]
        batch_size_actual = end_idx - start_idx
        
        # Dự đoán ban đầu (P^0)
        predictions = model(batch)
        predicted_classes = tf.argmax(predictions, axis=1)
        P0 = tf.gather(predictions, predicted_classes, axis=1, batch_dims=1)

        batch_expanded = tf.tile(batch[:, None, :], [1, n_features, 1])
        
        mask = np.ones((batch_size_actual, n_features, n_features), dtype=np.float32)
        for i in range(n_features):
            mask[:, i, i] = 0
        mask = tf.convert_to_tensor(mask, dtype=tf.float32)

        batch_modified = batch_expanded * mask
        batch_modified_flat = tf.reshape(batch_modified, [-1, n_features])

        predictions_modified = model(batch_modified_flat)
        predictions_modified = tf.reshape(predictions_modified, [batch_size_actual, n_features, -1])
        Pi = tf.gather(predictions_modified, predicted_classes[:, None], axis=2, batch_dims=1)

        P0_expanded = P0[:, None]
        P0_expanded = tf.tile(P0_expanded[:, :, None], [1, n_features, 1])
        delta = tf.abs(P0_expanded - Pi) / (P0_expanded + 1e-10)
        delta_sum = tf.reduce_sum(delta, axis=0)
        delta_sum = tf.squeeze(delta_sum, axis=-1)

        sens += delta_sum.numpy()
        
    sens /= n_samples
    return sens

def compute_ptb_classification(model, dataset, sigma=0.2, batch_size=128):
    n_samples, n_features = dataset.shape
    ptb = np.zeros(n_features, dtype=np.float32)

    dataset_tensor = tf.convert_to_tensor(dataset, dtype=tf.float32)
    num_batches = (n_samples + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch = dataset_tensor[start_idx:end_idx]
        batch_size_actual = end_idx - start_idx

        # Dự đoán ban đầu
        predictions = model(batch)
        predicted_classes = tf.argmax(predictions, axis=1)
        P0 = tf.gather(predictions, predicted_classes, axis=1, batch_dims=1) 

        base = tf.tile(batch[:, None, :], [1, n_features, 1]) 

        eye = tf.eye(n_features, batch_shape=[batch_size_actual]) * sigma
        x_plus = base + eye
        x_minus = base - eye

        x_all = tf.concat([x_plus, x_minus], axis=0)
        x_all = tf.reshape(x_all, [-1, n_features])

        preds_all = model(x_all)
        preds_all = tf.reshape(preds_all, [2, batch_size_actual, n_features, -1])

        # Lấy xác suất class đã chọn bằng one-hot
        num_classes = preds_all.shape[-1]
        one_hot = tf.one_hot(predicted_classes, depth=num_classes) 
        one_hot = tf.reshape(one_hot, [batch_size_actual, 1, num_classes]) 
        one_hot = tf.tile(one_hot, [1, n_features, 1]) 

        Pi_plus = tf.reduce_sum(preds_all[0] * one_hot, axis=-1)
        Pi_minus = tf.reduce_sum(preds_all[1] * one_hot, axis=-1)

        
        # Tính delta
        P0_expanded = tf.expand_dims(P0, axis=1) 
        delta = tf.abs(Pi_plus - P0_expanded) + tf.abs(Pi_minus - P0_expanded)

        # Cộng dồn
        delta_sum = tf.reduce_sum(delta, axis=0)
        ptb += delta_sum.numpy() / 2

    ptb /= n_samples
    return ptb

def compute_ptb_regression(model, dataset, sigma=0.2, batch_size=128):
    n_samples, n_features = dataset.shape
    ptb = np.zeros(n_features, dtype=np.float32)

    dataset_tensor = tf.convert_to_tensor(dataset, dtype=tf.float32)
    num_batches = (n_samples + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch = dataset_tensor[start_idx:end_idx]
        batch_size_actual = end_idx - start_idx

        # Dự đoán ban đầu
        predictions = model(batch)
        y0 = predictions

        base = tf.tile(batch[:, None, :], [1, n_features, 1]) 

        eye = tf.eye(n_features, batch_shape=[batch_size_actual]) * sigma
        x_plus = base + eye
        x_minus = base - eye

        x_all = tf.concat([x_plus, x_minus], axis=0)
        x_all = tf.reshape(x_all, [-1, n_features])

        preds_all = model(x_all)
        preds_all = tf.reshape(preds_all, [2, batch_size_actual, n_features, -1])

        y_plus = preds_all[0]
        y_minus = preds_all[1]

        # Mở rộng y0 để so sánh
        y0_expanded = tf.expand_dims(y0, axis=1)
        y0_expanded = tf.tile(y0_expanded, [1, n_features, 1])

        # Tính delta
        delta = tf.abs(y_plus - y0_expanded) + tf.abs(y_minus - y0_expanded)
        delta_sum = tf.reduce_sum(delta, axis=0)
        delta_sum = tf.reduce_mean(delta_sum, axis=-1)

        ptb += delta_sum.numpy() / 2

    ptb /= n_samples
    return ptb

def compute_cw(model):
    W1 = model.get_layer('hidden').get_weights()[0]
    W2 = model.get_layer('output').get_weights()[0]

    n_hidden, n_classes = W2.shape

    cw = np.zeros(W1.shape[0])
    for i in range(W1.shape[0]):
        for k in range(n_hidden):
            for j in range(n_classes):
                cw[i] += W1[i, k] * W2[k, j]

    return cw

__all__ = [
    'compute_sens',
    'compute_ptb_classification',
    'compute_ptb_regression',
    'compute_cw'
]