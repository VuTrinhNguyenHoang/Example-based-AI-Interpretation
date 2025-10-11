#!/usr/bin/env python3
"""
Regression Experiment - Example-based AI Interpretation
Converted from experiment_2/experiment.ipynb
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
# import argparse  # Not needed for fixed parameters
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from methods.compute_global import compute_sens, compute_ptb_regression as compute_ptb, compute_cw
from methods.compute_local import compute_llm_regression as compute_llm, compute_c_hp_regression as compute_c_hp
from utils.extractor import extract_raw_features

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def save_history(dataset, history, output_dir="results"):
    """Save training history plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss During Training')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset}_during_train.png'))
    plt.close()

def build_mlp(input_dim, hidden_units, output_dim=1):
    """Build MLP model for regression"""
    inputs = tf.keras.Input(shape=(input_dim,))
    hidden = tf.keras.layers.Dense(hidden_units, activation='relu', name='hidden')(inputs)
    output = tf.keras.layers.Dense(output_dim, name='output')(hidden)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

def rmse(mse):
    """Calculate RMSE from MSE"""
    return np.sqrt(mse)

def experiment(url_train, url_test, output_dir="results"):
    """Run regression experiment"""
    print("Loading data...")
    df_train = pd.read_csv(url_train)
    df_test = pd.read_csv(url_test)
    
    print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")

    target = df_train.columns[-1]
    X_train, X_test = df_train.drop(columns=[target]).values.astype(float), df_test.drop(columns=[target]).values.astype(float)
    y_train, y_test = df_train[target].values, df_test[target].values

    # Train MLP
    print("Training MLP...")
    model = build_mlp(X_train.shape[1], 128)
    history = model.fit(X_train, y_train, epochs=500, batch_size=128, validation_split=0.1, verbose=0)
    y_pred_test = model.predict(X_test, verbose=0)
    y_pred_train = model.predict(X_train, verbose=0)
    rmse_mlp = rmse(mean_squared_error(y_test, y_pred_test))
    print(f"MLP RMSE: {rmse_mlp:.4f}")
    
    # Train k-NN*
    print("Training k-NN*...")
    X_train_raw = extract_raw_features(model, X_train)
    X_test_raw = extract_raw_features(model, X_test)
    
    knn = KNeighborsRegressor(n_neighbors=1)
    knn.fit(X_train_raw, y_pred_train)
    y_pred_knn = knn.predict(X_test_raw)
    rmse_knn = rmse(mean_squared_error(y_test, y_pred_knn))
    rmse_mlp_knn = rmse(mean_squared_error(y_pred_test, y_pred_knn))
    
    metric = {
        'rmse': rmse_mlp,
        'rmse_k-NN*': rmse_knn,
        'rmse_mlp_knn': rmse_mlp_knn
    }

    # Train k-NN SENS
    print("Computing SENS and training k-NN SENS...")
    sens = compute_sens(model, X_train)
    scaling = np.sqrt(np.abs(sens))
    weights_train = X_train * scaling
    weights_test = X_test * scaling

    knn_weighted = KNeighborsRegressor(n_neighbors=1)
    knn_weighted.fit(weights_train, y_pred_train)
    y_pred_weighted = knn_weighted.predict(weights_test)
    rmse_weighted = rmse(mean_squared_error(y_test, y_pred_weighted))
    rmse_weighted_mlp = rmse(mean_squared_error(y_pred_test, y_pred_weighted))

    metric['rmse_sens'] = rmse_weighted
    metric['rmse_mlp_sens'] = rmse_weighted_mlp

    # Train k-NN PTB
    print("Computing PTB and training k-NN PTB...")
    ptb = compute_ptb(model, X_train)
    scaling = np.sqrt(np.abs(ptb))
    weights_train = X_train * scaling
    weights_test = X_test * scaling

    knn_weighted = KNeighborsRegressor(n_neighbors=1)
    knn_weighted.fit(weights_train, y_pred_train)
    y_pred_weighted = knn_weighted.predict(weights_test)
    rmse_weighted = rmse(mean_squared_error(y_test, y_pred_weighted))
    rmse_weighted_mlp = rmse(mean_squared_error(y_pred_test, y_pred_weighted))

    metric['rmse_ptb'] = rmse_weighted
    metric['rmse_mlp_ptb'] = rmse_weighted_mlp

    # Train k-NN CW
    print("Computing CW and training k-NN CW...")
    cw = compute_cw(model)
    scaling = np.sqrt(np.abs(cw))
    weights_train = X_train * scaling
    weights_test = X_test * scaling

    knn_weighted = KNeighborsRegressor(n_neighbors=1)
    knn_weighted.fit(weights_train, y_pred_train)
    y_pred_weighted = knn_weighted.predict(weights_test)
    rmse_weighted = rmse(mean_squared_error(y_test, y_pred_weighted))
    rmse_weighted_mlp = rmse(mean_squared_error(y_pred_test, y_pred_weighted))

    metric['rmse_cw'] = rmse_weighted
    metric['rmse_mlp_cw'] = rmse_weighted_mlp

    # Train k-NN LLM
    print("Computing LLM and training k-NN LLM...")
    weights_train = np.array([compute_llm(model, x) for x in tqdm(X_train, desc="Computing LLM for train")])
    weights_test = np.array([compute_llm(model, x) for x in tqdm(X_test, desc="Computing LLM for test")])

    knn_weighted = KNeighborsRegressor(n_neighbors=1)
    knn_weighted.fit(weights_train, y_pred_train)
    y_pred_weighted = knn_weighted.predict(weights_test)
    rmse_weighted = rmse(mean_squared_error(y_test, y_pred_weighted))
    rmse_weighted_mlp = rmse(mean_squared_error(y_pred_test, y_pred_weighted))

    metric['rmse_llm'] = rmse_weighted
    metric['rmse_mlp_llm'] = rmse_weighted_mlp

    # Train k-NN C-HP
    print("Computing C-HP and training k-NN C-HP...")
    weights_train = np.array([compute_c_hp(model, x) for x in tqdm(X_train, desc="Computing C-HP for train")])
    weights_test = np.array([compute_c_hp(model, x) for x in tqdm(X_test, desc="Computing C-HP for test")])

    knn_weighted = KNeighborsRegressor(n_neighbors=1)
    knn_weighted.fit(weights_train, y_pred_train)
    y_pred_weighted = knn_weighted.predict(weights_test)
    rmse_weighted = rmse(mean_squared_error(y_test, y_pred_weighted))
    rmse_weighted_mlp = rmse(mean_squared_error(y_pred_test, y_pred_weighted))

    metric['rmse_chp'] = rmse_weighted
    metric['rmse_mlp_chp'] = rmse_weighted_mlp
    
    return model, history, metric

def main():
    # Fixed parameters - same as original notebook
    data_dir = 'experiment_2/data_src'
    output_dir = 'results'
    dataset_names = ['bike_sharing', 'blog_feedback']
    
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = []

    for dataset in dataset_names:
        url_train = os.path.join(data_dir, f'{dataset}_train.csv')
        url_test = os.path.join(data_dir, f'{dataset}_test.csv')
        
        if not os.path.exists(url_train) or not os.path.exists(url_test):
            print(f"Warning: Data files for {dataset} not found. Skipping...")
            continue

        print(f"\n=== Running experiment: {dataset} ===")
        model, history, metric = experiment(url_train, url_test, output_dir)
        
        save_history(dataset, history, output_dir)
        metric['Dataset'] = dataset
        metrics.append(metric)

    # Save results
    df = pd.DataFrame(metrics)
    results_path = os.path.join(output_dir, 'regression_results.csv')
    df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    print("\nSample results:")
    print(df.head())

if __name__ == "__main__":
    main()