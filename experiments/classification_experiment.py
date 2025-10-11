#!/usr/bin/env python3
"""
Classification Experiment - Example-based AI Interpretation
Converted from experiment_1/experiment.ipynb
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import euclidean
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from pprint import pprint
import matplotlib.pyplot as plt
# import argparse  # Not needed for fixed parameters
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from methods.compute_global import compute_sens, compute_ptb_classification as compute_ptb, compute_cw
from methods.compute_local import compute_llm_classification as compute_llm, compute_c_hp_classification as compute_c_hp
from utils.extractor import extract_raw_features

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def save_history(dataset, history, output_dir="results"):
    """Save training history plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss during Training')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history.get('accuracy', history.history.get('acc')), label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy during Training')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset}_history.png'))
    plt.close()

def build_mlp(input_dim, hidden_units, num_classes):
    """Build MLP model"""
    inputs = tf.keras.Input(shape=(input_dim,), name="input")
    
    x = tf.keras.layers.Dense(hidden_units, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.01), 
                              name='hidden')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def experiment(url_train, url_test, n_neighbors, output_dir="results"):
    """Run classification experiment"""
    print(f"Running experiment with {n_neighbors} neighbors...")
    
    # Load data
    df_train = pd.read_csv(url_train)
    df_test = pd.read_csv(url_test)

    print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")

    target = df_train.columns[-1]
    X_train, X_test = df_train.drop(columns=[target]).values.astype(float), df_test.drop(columns=[target]).values.astype(float)
    y_train, y_test = df_train[target].values, df_test[target].values
    classes = np.unique(y_train)

    # Train MLP
    print("Training MLP...")
    model = build_mlp(X_train.shape[1], 64, len(classes))
    history = model.fit(X_train, y_train, epochs=30, batch_size=128, validation_split=0.1, verbose=0)
    y_pred_test = model.predict(X_test, verbose=0).argmax(axis=1)
    y_pred_train = model.predict(X_train, verbose=0).argmax(axis=1)
    acc = accuracy_score(y_test, y_pred_test)
    print(f"MLP Accuracy: {acc:.4f}")

    # Train k-NN*
    print("Training k-NN*...")
    X_train_raw = extract_raw_features(model, X_train)
    X_test_raw = extract_raw_features(model, X_test)
    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_raw, y_pred_train)
    y_pred_knn = knn.predict(X_test_raw)
    acc_knn = accuracy_score(y_test, y_pred_knn)
    agreement = accuracy_score(y_pred_test, y_pred_knn)

    metric = {
        'k_neighbors': n_neighbors,
        'acc': acc,
        'acc_k-NN*': acc_knn,
        'agreement_k-NN*': agreement
    }

    # Train k-NN SENS
    print("Computing SENS and training k-NN SENS...")
    sens = compute_sens(model, X_train)
    scaling = np.sqrt(np.abs(sens))
    weights_train = X_train * scaling
    weights_test = X_test * scaling

    knn_weighted = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_weighted.fit(weights_train, y_pred_train)
    y_pred_weighted = knn_weighted.predict(weights_test)
    acc_weighted = accuracy_score(y_test, y_pred_weighted)
    agreement_weighted = accuracy_score(y_pred_test, y_pred_weighted)

    metric['acc_sens'] = acc_weighted
    metric['agreement_sens'] = agreement_weighted

    # Train k-NN PTB
    print("Computing PTB and training k-NN PTB...")
    ptb = compute_ptb(model, X_train)
    scaling = np.sqrt(np.abs(ptb))
    weights_train = X_train * scaling
    weights_test = X_test * scaling

    knn_weighted = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_weighted.fit(weights_train, y_pred_train)
    y_pred_weighted = knn_weighted.predict(weights_test)
    acc_weighted = accuracy_score(y_test, y_pred_weighted)
    agreement_weighted = accuracy_score(y_pred_test, y_pred_weighted)

    metric['acc_ptb'] = acc_weighted
    metric['agreement_ptb'] = agreement_weighted

    # Train k-NN CW
    print("Computing CW and training k-NN CW...")
    cw = compute_cw(model)
    scaling = np.sqrt(np.abs(cw))
    weights_train = X_train * scaling
    weights_test = X_test * scaling

    knn_weighted = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_weighted.fit(weights_train, y_pred_train)
    y_pred_weighted = knn_weighted.predict(weights_test)
    acc_weighted = accuracy_score(y_test, y_pred_weighted)
    agreement_weighted = accuracy_score(y_pred_test, y_pred_weighted)

    metric['acc_cw'] = acc_weighted
    metric['agreement_cw'] = agreement_weighted

    # Train k-NN LLM
    print("Computing LLM and training k-NN LLM...")
    weights_train = np.array([compute_llm(model, x) for x in tqdm(X_train, desc="Computing LLM for train")])
    weights_test = np.array([compute_llm(model, x) for x in tqdm(X_test, desc="Computing LLM for test")])

    knn_weighted = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_weighted.fit(weights_train, y_pred_train)
    y_pred_weighted = knn_weighted.predict(weights_test)
    acc_weighted = accuracy_score(y_test, y_pred_weighted)
    agreement_weighted = accuracy_score(y_pred_test, y_pred_weighted)

    metric['acc_llm'] = acc_weighted
    metric['agreement_llm'] = agreement_weighted

    # Train k-NN C-HP
    print("Computing C-HP and training k-NN C-HP...")
    weights_train = np.array([compute_c_hp(model, x) for x in tqdm(X_train, desc="Computing C-HP for train")])
    weights_test = np.array([compute_c_hp(model, x) for x in tqdm(X_test, desc="Computing C-HP for test")])

    knn_weighted = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_weighted.fit(weights_train, y_pred_train)
    y_pred_weighted = knn_weighted.predict(weights_test)
    acc_weighted = accuracy_score(y_test, y_pred_weighted)
    agreement_weighted = accuracy_score(y_pred_test, y_pred_weighted)

    metric['acc_chp'] = acc_weighted
    metric['agreement_chp'] = agreement_weighted

    return model, history, metric

def main():
    # Fixed parameters - same as original notebook
    data_dir = 'experiment_1/data_src'
    output_dir = 'results'
    dataset_names = ['bank_marketing', 'breast_cancer', 'nursery']
    k_neighbors_list = [1, 2, 3, 4, 5]
    
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = []
    
    for n_neighbors in k_neighbors_list:
        for dataset in dataset_names:
            url_train = os.path.join(data_dir, f'{dataset}_train.csv')
            url_test = os.path.join(data_dir, f'{dataset}_test.csv')
            
            if not os.path.exists(url_train) or not os.path.exists(url_test):
                print(f"Warning: Data files for {dataset} not found. Skipping...")
                continue
        
            print(f"\n=== Running experiment: {dataset} with k={n_neighbors} ===")
            model, history, metric = experiment(url_train, url_test, n_neighbors, output_dir)
            
            save_history(dataset, history, output_dir)
            metric['Dataset'] = dataset
            metrics.append(metric)
    
    # Save results
    df = pd.DataFrame(metrics)
    results_path = os.path.join(output_dir, 'classification_results.csv')
    df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    print("\nSample results:")
    print(df.head())

if __name__ == "__main__":
    main()