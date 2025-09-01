# 🔤🧠Transformer-Based Turkish Words' Root Finder🪵

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)

A deep learning model that finds the roots of Turkish words using Transformer architecture. The model employs sequence-to-sequence (seq2seq) approach and supports both greedy decoding and beam search algorithms.

## ✨ Features

- 🤖 **Transformer Architecture**: Powerful performance with modern attention mechanism
- 🔤 **Character-level Tokenization**: Supports Turkish characters (ç, ğ, ı, ö, ş, ü)
- 🔎 **Dual Decoding**: Greedy and Beam Search algorithms
- 🎨 **Label Smoothing**: Advanced loss function to prevent overfitting
- ⏹️ **Early Stopping & LR Scheduling**: Callbacks for optimal training
- 📊 **Comprehensive EDA**: Data analysis and visualization

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

## 📁 Project Structure
