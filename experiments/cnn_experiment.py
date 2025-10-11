#!/usr/bin/env python3
"""
CNN Experiment - Example-based AI Interpretation
Converted from experiment_3/experiment.ipynb and experiment_3/experiment_dog_cat.ipynb
"""

import tensorflow as tf
from tensorflow.keras import regularizers, Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, GlobalAveragePooling2D

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
# import argparse  # Not needed for fixed parameters
import os
import sys
import cv2

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_data(dataset_name, data_fraction=1):
    """Load dataset (MNIST, Fashion-MNIST, CIFAR-10)"""
    if dataset_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        num_channels = 1
        input_shape = (28, 28, num_channels)
        num_classes = 10
    elif dataset_name == 'mnist-fashion':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        num_channels = 1
        input_shape = (28, 28, num_channels)
        num_classes = 10
    elif dataset_name == 'cifar-10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        num_channels = 3
        input_shape = (32, 32, num_channels)
        num_classes = 10
    else:
        raise ValueError(f"Don't support Dataset {dataset_name}")

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    if data_fraction < 1:
        num_train = len(x_train)
        num_test = len(x_test)
        train_indices = np.random.choice(num_train, int(num_train * data_fraction), replace=False)
        test_indices = np.random.choice(num_test, int(num_test * data_fraction), replace=False)
        x_train = x_train[train_indices]
        y_train = y_train[train_indices]
        x_test = x_test[test_indices]
        y_test = y_test[test_indices]
    
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.2, random_state=42)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), num_channels, input_shape, num_classes

def build_cnn(input_shape, num_classes, model_name):
    """Build CNN model"""
    inputs = Input(shape=input_shape, name='input')
    x = Conv2D(32, (3, 3), activation='relu', name='conv1')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', name='conv2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, (3, 3), activation='relu', name='conv3')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', name='conv4')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, (3, 3), activation='relu', name='conv5')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.4)(x)

    x = Flatten()(x) if model_name == 'flatten' else GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax', name='output')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def compute_c_hp(model, x, last_conv_layer_name):
    """Compute C-HP (Class-specific Heatmap Projection) features"""
    x_tensor = tf.convert_to_tensor(x[None, ...], dtype=tf.float32)
    
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.outputs]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x_tensor)
        predicted_class = np.argmax(predictions[0].numpy())
        
        class_output = predictions[0][0][predicted_class]
    
    gradients = tape.gradient(class_output, conv_outputs)
    if gradients is None:
        raise ValueError("Gradients is None. Check model layer connections!")
    
    c_hp = (gradients * conv_outputs)[0]
    
    c_hp_flat = tf.reshape(c_hp, [-1])
    return c_hp_flat.numpy()

def plot_predictions(model, x, y, labels, num_images=10, output_dir="results"):
    """Plot model predictions"""
    pred = model.predict(x, verbose=0).argmax(axis=1)

    plt.figure(figsize=(12, 4))
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x[i])
        true_label = labels[y[i]]
        pred_label = labels[pred[i]]
        color = 'green' if true_label == pred_label else 'red'
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color) 
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions.png'))
    plt.close()

def twin_system_test(dataset_name, model_name, cnn, knn, x_train, y_train, x_test, y_test, labels, num_channels, output_dir="results"):
    """Test Twin System visualization"""
    rand_idx = np.random.randint(0, x_test.shape[0])
    test_img = x_test[rand_idx]
    cnn_pred = np.argmax(cnn.predict(test_img[None, ...], verbose=0), axis=1)[0]

    chp = compute_c_hp(cnn, test_img, 'conv5')
    knn_index = knn.kneighbors(chp.reshape(1, -1), n_neighbors=1, return_distance=False)[0][0]
    retrieved_img = x_train[knn_index]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(np.squeeze(test_img), cmap='gray' if num_channels == 1 else None)
    color = 'green' if y_test[rand_idx] == cnn_pred else 'red'
    plt.title(f'Test Image\nCNN predicted: {labels[cnn_pred]}\nTrue: {labels[y_test[rand_idx]]}', color=color)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(np.squeeze(retrieved_img), cmap='gray' if num_channels == 1 else None)
    plt.title(f'Nearest Image\nLabels: {labels[y_train[knn_index]]}')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'twin_system_{model_name}_{dataset_name}.png'))
    plt.close()

def train(model_name, dataset_name, model, x_train, y_train, x_valid, y_valid, epochs=50, batch_size=64, output_dir="results"):
    """Train CNN model"""
    print(f"Training {model_name} on {dataset_name}...")
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, 
                       validation_data=(x_valid, y_valid), verbose=0)

    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(os.path.join(output_dir, f'{dataset_name}_{model_name}_history.png'))
    plt.close()

def experiment(dataset_name, model_name, labels, epochs=50, data_fraction=1, output_dir="results"):
    """Run CNN experiment"""
    print(f"\n=== Running experiment: {dataset_name} with {model_name} ===")
    
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test), num_channels, input_shape, num_classes = load_data(dataset_name, data_fraction)
    model = build_cnn(input_shape, num_classes, model_name)

    # Train CNN
    train(model_name, dataset_name, model, x_train, y_train, x_valid, y_valid, epochs=epochs, output_dir=output_dir)
    
    # Save model
    model_path = os.path.join(output_dir, f'{model_name}_{dataset_name}.h5')
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    y_train_pred = model.predict(x_train, verbose=0).argmax(axis=1)
    y_test_pred = model.predict(x_test, verbose=0).argmax(axis=1)
    acc = accuracy_score(y_test, y_test_pred)

    print(f'Accuracy {model_name} with {dataset_name}: {acc:.4f}')

    # Plot random 10 images
    indices = np.random.choice(len(x_test), 10, replace=False)
    plot_predictions(model, x_test[indices], y_test[indices], labels, output_dir=output_dir)

    # Train k-NN*
    print("Training k-NN* with raw features...")
    feature_extractor = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer('conv5').output, model.outputs]
    )
    x_train_features = feature_extractor.predict(x_train, verbose=0)[0].reshape(len(x_train), -1)
    x_test_features = feature_extractor.predict(x_test, verbose=0)[0].reshape(len(x_test), -1)
    
    knn_star = KNeighborsClassifier(n_neighbors=1)
    knn_star.fit(x_train_features, y_train_pred)
    y_test_knn_star = knn_star.predict(x_test_features)

    accuracy_star = accuracy_score(y_test, y_test_knn_star)
    agreement_star = accuracy_score(y_test_pred, y_test_knn_star)
    
    print("Agreement between CNN and k-NN (raw features):", agreement_star)
    print("Accuracy of k-NN (raw features) vs true labels:", accuracy_star)
    
    # Train k-NN with C-HP
    print("Computing C-HP features and training k-NN C-HP...")
    C_train = np.array([compute_c_hp(model, x, 'conv5') for x in tqdm(x_train, desc="Computing C-HP for train")])
    C_test = np.array([compute_c_hp(model, x, 'conv5') for x in tqdm(x_test, desc="Computing C-HP for test")])
    knn_chp = KNeighborsClassifier(n_neighbors=1)
    knn_chp.fit(C_train, y_train_pred)
    y_test_knn_chp = knn_chp.predict(C_test)

    accuracy_chp = accuracy_score(y_test, y_test_knn_chp)
    agreement_chp = accuracy_score(y_test_pred, y_test_knn_chp)

    print('Agreement between CNN and k-NN C-HP:', agreement_chp)
    print('Accuracy of k-NN C-HP vs true labels:', accuracy_chp)

    # Test Twin-System
    twin_system_test(dataset_name, model_name, model, knn_chp, x_train, y_train, x_test, y_test, labels, num_channels, output_dir)

    return {
        'Dataset': dataset_name,
        'Model': model_name,
        'acc': acc,
        'acc k-NN*': accuracy_star,
        'agreement k-NN*': agreement_star,
        'acc C-HP': accuracy_chp,
        'agreement C-HP': agreement_chp
    }

def main():
    # Fixed parameters - same as original notebook
    output_dir = 'results'
    dataset_names = ['mnist', 'mnist-fashion', 'cifar-10']
    model_names = ['gap', 'flatten']
    epochs = 50
    data_fraction = 0.5
    
    os.makedirs(output_dir, exist_ok=True)
    
    labels = {
        'mnist': [str(i) for i in range(10)],
        'mnist-fashion': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
        'cifar-10': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    }

    all_results = []
    for model in model_names:
        results = []
        for dataset in dataset_names:
            result = experiment(dataset, model, labels[dataset], epochs, data_fraction, output_dir)
            results.append(result)
            all_results.append(result)

        # Save results for each model
        df = pd.DataFrame(results)
        model_results_path = os.path.join(output_dir, f'results_{model}.csv')
        df.to_csv(model_results_path, index=False)
        print(f"Results for {model} saved to: {model_results_path}")

    # Save all results
    df_all = pd.DataFrame(all_results)
    all_results_path = os.path.join(output_dir, 'cnn_all_results.csv')
    df_all.to_csv(all_results_path, index=False)
    print(f"\nAll results saved to: {all_results_path}")
    print("\nSample results:")
    print(df_all.head())

if __name__ == "__main__":
    main()