# 🔤🧠Transformer-Based Turkish Words' Root Finder🪵

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)

A deep learning model that finds the roots of Turkish words using Transformer architecture. The model employs sequence-to-sequence (seq2seq) approach and supports both greedy decoding and beam search algorithms.

## ✨ Features

- 🤖 **Transformer Architecture**: Powerful performance with modern attention mechanism
- 🔤 **Character-level Tokenization**: Supports Turkish characters (ç, ğ, ı, ö, ş, ü)
- �� **Dual Decoding**: Greedy and Beam Search algorithms
- 🎨 **Label Smoothing**: Advanced loss function to prevent overfitting
- ⏹️ **Early Stopping & LR Scheduling**: Callbacks for optimal training
- 📊 **Comprehensive EDA**: Data analysis and visualization

## 🛠️ Technologies & Libraries Used

### 🧠 **Deep Learning Framework**
- **TensorFlow 2.10+**: Core deep learning framework
- **Keras**: High-level neural network API
- **GPU Support**: CUDA acceleration for faster training

### 📊 **Data Processing & Analysis**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning utilities (train_test_split)

### �� **Visualization**
- **Matplotlib**: Plotting and visualization
- **Seaborn**: Statistical data visualization
- **Custom Plot Themes**: Professional styling with whitegrid theme

### 🔧 **Model Architecture Components**
- **MultiHeadAttention**: Self-attention and cross-attention mechanisms
- **LayerNormalization**: Stable training with residual connections
- **Embedding Layers**: Character-level embeddings with positional encoding
- **Dense Layers**: Feed-forward networks
- **Dropout**: Regularization technique

### 🎯 **Training Features**
- **ModelCheckpoint**: Save best model weights
- **EarlyStopping**: Prevent overfitting
- **ReduceLROnPlateau**: Adaptive learning rate scheduling
- **Custom Loss Function**: Sparse categorical cross-entropy with label smoothing
- **Teacher Forcing**: Training strategy for seq2seq models

### 🔤 **Text Processing**
- **Character-level Tokenization**: Custom tokenizer for Turkish characters
- **Vocabulary Management**: Dynamic vocabulary building
- **Padding & Masking**: Sequence padding and attention masking
- **Special Tokens**: START, END, PAD, UNK tokens

## 📈 Model Performance

- **Total Parameters**: 934,691 (3.57 MB)
- **Final Validation Loss**: 0.7428
- **Validation Accuracy**: 43.32%
- **Training Epochs**: 33 (with early stopping)

## ⚙️ Technical Details

### 🏗️ Model Architecture
- **Encoder Layers**: 2
- **Decoder Layers**: 2
- **Embedding Dimension**: 128
- **Attention Heads**: 4
- **Feed Forward Dimension**: 512
- **Dropout Rate**: 0.1

### ⚡🛠️ Hyperparameters
- **Batch Size**: 64
- **Learning Rate**: 5e-4 (with ReduceLROnPlateau)
- **Label Smoothing**: 0.1
- **Beam Size**: 3
- **Max Encoder Length**: 34
- **Max Decoder Length**: 13

### 📖 Vocabulary Details
- **Total Characters**: 35 (including special tokens)
- **Turkish Characters**: ç, ğ, ı, ö, ş, ü
- **Special Tokens**: <PAD>, <UNK>, <, >
- **Vocabulary Size**: 35

### 🔄 Callbacks
- **ModelCheckpoint**: Save best weights
- **EarlyStopping**: Prevent overfitting
- **ReduceLROnPlateau**: Automatically adjust learning rate

## �� Dataset

The model is trained on 9,154 Turkish word pairs:
- **Total Words**: 9,154
- **Average Word Length**: 8.19 characters
- **Average Root Length**: 4.71 characters
- **Words without Suffix**: 19.03%

### 📝 Data Examples
| Word | Root | Suffix |
|------|------|--------|
| alıyorum | al | ıyorum |
| görmek | gör | mek |
| şiddetli | şiddet | li |
| yumurtalar | yumurta | lar |


## 🧪 Test Results

### ✅ Sample Predictions
| Word | True Root | Predicted | Correct |
|------|-----------|-----------|---------|
| evimdeyken | ev | ev | ✅ |
| kitapta | kitap | kitap | ✅ |
| görüyordum | gör | gör | ✅ |
| koşacaklarmış | koş | koş | ✅ |
| bulutlardayım | bulut | bulut | ✅ |
| tevekkül | tevekkül | tevekkül | ✅ |
| cömert | cömert | cömert | ✅ |
| etkilendim | etkilen | etkilen | ✅ |
| önderimiz | önder | önder | ✅ |
| anlamlandıramadıklarımdan | anlam | anlam | ✅ |

## 📊 Performance Analysis

### 📈 Loss and Accuracy Charts
During model training:
- **Training Loss**: 0.6894 (final)
- **Validation Loss**: 0.7428 (final)
- **Training Accuracy**: 43.95% (final)
- **Validation Accuracy**: 43.32% (final)

### 📉 Learning Rate Schedule
- Start: 5e-4
- Epoch 21: 2.5e-4 (ReduceLROnPlateau)
- Epoch 29: 1.25e-4
- Epoch 32: 6.25e-5

## 🎨 Visualizations

The project includes the following visualizations:
- 📏 **Length Distributions**: Word/root/suffix length histograms
- 🔝 **Frequency Analysis**: Top roots and suffixes bar charts
- 📊 **Suffix Buckets**: Pie chart of suffix length distribution
- �� **Training Metrics**: Loss/accuracy graphs during training
- �� **Learning Rate**: Learning rate change graph


## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
