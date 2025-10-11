# A letter from the black-box

This is the source code of a paper submitted to PeerJ Computer Science journal

---

## Table of Contents
A letter from the black box: Interpreting artificial intelligence through example-based Twin systems
1. [System Requirements](#system-requirements)
2. [Environment Setup](#environment-setup)
3. [Dataset](#dataset)
4. [Usage Guide](#usage-guide)
   - [Quick Start](#quick-start)
   - [Individual Experiments](#individual-experiments)
   - [Custom Parameters](#custom-parameters)
5. [Experiments Description](#experiments-description)
6. [Results](#results)

---

## System Requirements
- **Python**: 3.10
- **Dependencies**: Listed in `requirements.txt`

---

## Environment Setup

### 1. Clone the Repository
Open your terminal and run:
```bash
git clone https://github.com/VuTrinhNguyenHoang/Example-based-AI-Interpretation.git
cd Example-based-AI-Interpretation
```

### 2. Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Dataset

The datasets used in our experiments are publicly available from the following sources:

### Tabular Data (Experiments 1 & 2)
- **Bank Marketing, Breast Cancer, Nursery**: [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset)
- **Bike Sharing, Blog Feedback**: [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset)

### Image Data (Experiment 3)
- **MNIST**: [Hugging Face - ylecun/mnist](https://huggingface.co/datasets/ylecun/mnist)
- **MNIST-Fashion**: [Hugging Face - visual-layer/fashion-mnist-vl-enriched](https://huggingface.co/datasets/visual-layer/fashion-mnist-vl-enriched)
- **CIFAR-10**: [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Dogs vs Cats**: [Kaggle Competition](https://www.kaggle.com/c/dogs-vs-cats)

### Text Data (Experiment 4)
- **AG News**: [Hugging Face - sh0416/ag_news](https://huggingface.co/datasets/sh0416/ag_news)
- **IMDB**: [Stanford AI Lab](https://ai.stanford.edu/~amaas/data/sentiment/)

## Usage Guide

### Quick Start

To run all experiments with default settings:
```bash
python main.py
```

This will execute all four experiments sequentially and save results to the `results/` directory.

### Individual Experiments

#### 1. Classification Experiment (MLP + k-NN variants)
```bash
python experiments/classification_experiment.py
```

#### 2. Regression Experiment (MLP + k-NN variants)
```bash
python experiments/regression_experiment.py
```

#### 3. CNN Experiment (Image classification with twin systems)
```bash
python experiments/cnn_experiment.py
```

#### 4. BERT Experiment (Text classification with interpretability)
```bash
python experiments/bert_experiment.py
```

---