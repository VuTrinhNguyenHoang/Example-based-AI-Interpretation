import tensorflow as tf
from tqdm.auto import tqdm
import numpy as np

def extract_raw_features(model, data, layer_name='hidden', batch_size=128):
    feature_model = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer(layer_name).output)
    
    data_tensor = tf.convert_to_tensor(data, dtype=tf.float32)
    n_samples = data.shape[0]
    features = []

    for start_idx in tqdm(range(0, n_samples, batch_size), desc="Processing batches"):
        end_idx = min(start_idx + batch_size, n_samples)
        batch = data_tensor[start_idx:end_idx]
        batch_features = feature_model(batch)
        features.append(batch_features.numpy())
    
    return np.concatenate(features, axis=0)