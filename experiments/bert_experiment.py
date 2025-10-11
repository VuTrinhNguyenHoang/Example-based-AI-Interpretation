#!/usr/bin/env python3
"""
BERT Experiment - Example-based AI Interpretation
Converted from experiment_4/experiment.ipynb
"""

import gc
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import BertTokenizer, TFBertModel
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
# import argparse  # Not needed for fixed parameters
import os
import sys

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Clear memory
tf.keras.backend.clear_session()
gc.collect()

class BertClassifier(tf.keras.Model):
    """BERT Classifier model"""
    def __init__(self, bert_model, num_classes=4):
        super(BertClassifier, self).__init__()
        self.bert = bert_model
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, **kwargs):
        outputs = self.bert(**inputs)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)

def tokenize_texts(texts, tokenizer, max_length=256):
    """Tokenize texts using BERT tokenizer"""
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='tf'
    )

def load_ag_news_data(train_size=40000, valid_size=5000, test_size=5000):
    """Load AG News dataset"""
    print("Loading AG News dataset...")
    dataset = load_dataset('ag_news')
    
    # Get train, valid and test datasets
    train_dataset = dataset['train'].select(range(train_size))
    valid_dataset = dataset['train'].select(range(train_size, train_size + valid_size))
    test_dataset = dataset['test'].select(range(test_size))
    
    return train_dataset, valid_dataset, test_dataset

def compute_c_hp(texts, class_indices, bert_classifier, tokenizer, batch_size=8):
    """Compute C-HP features"""
    bert_model = bert_classifier.bert
    embeddings_layer = bert_model.get_input_embeddings()

    c_hp_features = []
    num_texts = len(texts)

    # Process texts in batches
    for i in tqdm(range(0, num_texts, batch_size), desc="Computing C-HP features"):
        batch_texts = texts[i:i + batch_size]
        batch_indices = class_indices[i:i + batch_size]
        
        # Tokenize the batch
        inputs = tokenizer(
            batch_texts,
            return_tensors='tf',
            padding=True,
            truncation=True,
            max_length=256
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Convert input_ids to embeddings
        embeddings = embeddings_layer(input_ids)
        embeddings = tf.cast(embeddings, tf.float32)

        # Use GradientTape to compute gradients
        with tf.GradientTape() as tape:
            tape.watch(embeddings)
            # Forward pass with inputs_embeds
            outputs = bert_model({
                "inputs_embeds": embeddings,
                "attention_mask": attention_mask
            })
            cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            preds = bert_classifier.classifier(cls_output)
            # Compute loss for the specified class_idx for the entire batch
            batch_indices_tensor = tf.constant(batch_indices, dtype=tf.int32)
            batch_indices_range = tf.range(tf.shape(preds)[0])
            selected_preds = tf.gather_nd(preds, tf.stack([batch_indices_range, batch_indices_tensor], axis=1))
            loss = tf.reduce_mean(selected_preds)

        # Compute gradients with respect to [CLS] output
        grads = tape.gradient(loss, cls_output)
        if grads is None:
            raise ValueError("Gradients is None. Check if the model is trainable or if cls_output is properly watched.")

        # Compute C-HP: Element-wise multiplication of gradients and [CLS] embedding
        c_hp = grads * cls_output
        c_hp_flat = tf.reshape(c_hp, [tf.shape(c_hp)[0], -1])
        c_hp_features.append(c_hp_flat.numpy())

    # Stack all features into a single array
    return np.vstack(c_hp_features)

def compute_raw_features(texts, class_indices, bert_classifier, tokenizer, batch_size=8):
    """Compute raw features (CLS embeddings)"""
    bert_model = bert_classifier.bert
    embeddings_layer = bert_model.get_input_embeddings()

    raw_features = []
    num_texts = len(texts)

    # Process texts in batches
    for i in tqdm(range(0, num_texts, batch_size), desc="Computing raw features"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize the batch
        inputs = tokenizer(
            batch_texts,
            return_tensors='tf',
            padding=True,
            truncation=True,
            max_length=256
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Convert input_ids to embeddings
        embeddings = embeddings_layer(input_ids)
        embeddings = tf.cast(embeddings, tf.float32)

        # Forward pass to get CLS output
        outputs = bert_model({
            "inputs_embeds": embeddings,
            "attention_mask": attention_mask
        })
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            
        cls_flat = tf.reshape(cls_output, [tf.shape(cls_output)[0], -1])
        raw_features.append(cls_flat.numpy())

    # Stack all features into a single array
    return np.vstack(raw_features)

def predict_with_bert(bert_classifier, texts, tokenizer, batch_size=8):
    """Make predictions with BERT"""
    predictions = []
    num_texts = len(texts)

    # Process texts in batches
    for i in tqdm(range(0, num_texts, batch_size), desc="Predicting with BERT"):
        batch_texts = texts[i:i + batch_size]

        # Tokenize the batch
        inputs = tokenizer(
            batch_texts,
            return_tensors='tf',
            padding=True,
            truncation=True,
            max_length=256
        )

        # Predict for current batch
        batch_predictions = bert_classifier(dict(inputs))
        batch_labels = np.argmax(batch_predictions, axis=1)
        predictions.append(batch_labels)

    # Concatenate all predictions into a single array
    return np.concatenate(predictions)

def experiment(train_size=40000, valid_size=5000, test_size=5000, epochs=2, batch_size=8, output_dir="results"):
    """Run BERT experiment"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    train_dataset, valid_dataset, test_dataset = load_ag_news_data(train_size, valid_size, test_size)
    
    # Initialize tokenizer and model
    print("Initializing BERT tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    bert_classifier = BertClassifier(bert_model, num_classes=4)
    
    bert_classifier.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    # Prepare texts and labels
    X_train_texts = [item['text'] for item in train_dataset]
    X_valid_texts = [item['text'] for item in valid_dataset]
    X_test_texts = [item['text'] for item in test_dataset]
    
    train_encodings = tokenize_texts(X_train_texts, tokenizer)
    valid_encodings = tokenize_texts(X_valid_texts, tokenizer)
    test_encodings = tokenize_texts(X_test_texts, tokenizer)
    
    train_labels = tf.convert_to_tensor([item['label'] for item in train_dataset], dtype=tf.int32)
    valid_labels = tf.convert_to_tensor([item['label'] for item in valid_dataset], dtype=tf.int32)
    test_labels = tf.convert_to_tensor([item['label'] for item in test_dataset], dtype=tf.int32)
    
    # Create TensorFlow datasets
    train_tf_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        train_labels
    ))
    
    valid_tf_dataset = tf.data.Dataset.from_tensor_slices((
        dict(valid_encodings),
        valid_labels
    ))
    
    test_tf_dataset = tf.data.Dataset.from_tensor_slices((
        dict(test_encodings),
        test_labels
    ))
    
    # Batch the datasets
    train_tf_dataset = train_tf_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    valid_tf_dataset = valid_tf_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_tf_dataset = test_tf_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Train model
    print("Training BERT classifier...")
    history = bert_classifier.fit(
        train_tf_dataset,
        validation_data=valid_tf_dataset,
        epochs=epochs,
        verbose=1
    )
    
    # Save model
    model_path = os.path.join(output_dir, 'bert_classifier')
    bert_classifier.save_weights(model_path)
    print(f"Model weights saved to: {model_path}")
    
    # Make predictions
    print("Making predictions...")
    y_train_pred = predict_with_bert(bert_classifier, X_train_texts, tokenizer, batch_size)
    y_test_pred = predict_with_bert(bert_classifier, X_test_texts, tokenizer, batch_size)
    
    bert_acc = accuracy_score(test_labels, y_test_pred)
    print(f'BERT Accuracy: {bert_acc:.4f}')
    
    # Compute C-HP features
    print("Computing C-HP features...")
    c_hp_features_train = compute_c_hp(
        texts=X_train_texts,
        class_indices=y_train_pred,
        bert_classifier=bert_classifier,
        tokenizer=tokenizer,
        batch_size=batch_size
    )
    
    c_hp_features_test = compute_c_hp(
        texts=X_test_texts,
        class_indices=y_test_pred,
        bert_classifier=bert_classifier,
        tokenizer=tokenizer,
        batch_size=batch_size
    )
    
    # Train k-NN with C-HP features
    print("Training k-NN with C-HP features...")
    knn_chp = KNeighborsClassifier(n_neighbors=1)
    knn_chp.fit(c_hp_features_train, y_train_pred)
    y_knn_chp_pred = knn_chp.predict(c_hp_features_test)
    
    chp_acc = accuracy_score(test_labels, y_knn_chp_pred)
    chp_agreement = accuracy_score(y_test_pred, y_knn_chp_pred)
    
    print(f'k-NN C-HP Accuracy: {chp_acc:.4f}')
    print(f'k-NN C-HP Agreement: {chp_agreement:.4f}')
    
    # Compute raw features
    print("Computing raw features...")
    raw_features_train = compute_raw_features(
        texts=X_train_texts,
        class_indices=y_train_pred,
        bert_classifier=bert_classifier,
        tokenizer=tokenizer,
        batch_size=batch_size
    )
    
    raw_features_test = compute_raw_features(
        texts=X_test_texts,
        class_indices=y_test_pred,
        bert_classifier=bert_classifier,
        tokenizer=tokenizer,
        batch_size=batch_size
    )
    
    # Train k-NN with raw features
    print("Training k-NN with raw features...")
    knn_raw = KNeighborsClassifier(n_neighbors=1)
    knn_raw.fit(raw_features_train, y_train_pred)
    y_knn_raw_pred = knn_raw.predict(raw_features_test)
    
    raw_acc = accuracy_score(test_labels, y_knn_raw_pred)
    raw_agreement = accuracy_score(y_test_pred, y_knn_raw_pred)
    
    print(f'k-NN Raw Accuracy: {raw_acc:.4f}')
    print(f'k-NN Raw Agreement: {raw_agreement:.4f}')
    
    # Test similarity search
    print("\nTesting similarity search...")
    sample_id = 0
    sample_text = X_test_texts[sample_id]
    sample_classIdx = y_test_pred[sample_id]
    true_label = test_labels[sample_id].numpy()
    
    sample_chp = compute_c_hp(
        texts=[sample_text],
        class_indices=[sample_classIdx],
        bert_classifier=bert_classifier,
        tokenizer=tokenizer,
        batch_size=1
    )
    
    distances, indices = knn_chp.kneighbors(sample_chp)
    
    print(f"\nTest sample: {sample_text}")
    print(f"Prediction: {knn_chp.predict(sample_chp)[0]}")
    print("Similar cases (indices):", indices[0])
    print("Distances:", distances[0])
    print("\nSimilar texts:")
    for idx in indices[0][:3]:  # Show top 3 similar texts
        print(f"- {X_train_texts[idx]} (Label: {train_labels[idx]})")
    
    # Save results
    results = {
        'bert_accuracy': bert_acc,
        'knn_chp_accuracy': chp_acc,
        'knn_chp_agreement': chp_agreement,
        'knn_raw_accuracy': raw_acc,
        'knn_raw_agreement': raw_agreement,
        'train_size': train_size,
        'test_size': test_size,
        'epochs': epochs
    }
    
    results_df = pd.DataFrame([results])
    results_path = os.path.join(output_dir, 'bert_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    return results

def main():
    # Fixed parameters - same as original notebook
    output_dir = 'results'
    train_size = 40000  # Same as in original notebook
    valid_size = 5000
    test_size = 5000
    epochs = 2
    batch_size = 8
    
    print("=== Starting BERT Experiment ===")
    results = experiment(
        train_size=train_size,
        valid_size=valid_size,
        test_size=test_size,
        epochs=epochs,
        batch_size=batch_size,
        output_dir=output_dir
    )
    
    print("\n=== Experiment completed ===")
    print("Results:")
    for key, value in results.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()