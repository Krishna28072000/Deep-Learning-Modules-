# 🚀 Deep Learning Modules

This repository contains implementations of various Deep Learning models using **TensorFlow (Keras)** and **PyTorch**.  
It is designed to demonstrate core concepts of Neural Networks, Computer Vision, NLP, and Generative Models.

---

## 📂 Projects Included

### 🔹 1. ANN - MNIST Digit Classification
- Built using a **Fully Connected Neural Network**
- Dataset: MNIST (handwritten digits)
- Key Concepts:
  - Data preprocessing
  - Dense layers
  - Softmax classification
- Accuracy: ~97–98%

---

### 🔹 2. CNN - MNIST Image Classification
- Uses **Convolutional Neural Network**
- Extracts spatial features from images
- Key Layers:
  - Conv2D
  - MaxPooling
  - Flatten
- Better performance than ANN

---

### 🔹 3. RNN - Character-Level Text Generator
- Generates text **character by character**
- Uses **SimpleRNN**
- Learns sequence patterns from input text
- Demonstrates basic NLP concepts

---

### 🔹 4. GRU - Time Series Forecasting
- Predicts **next day temperature**
- Uses **GRU (Gated Recurrent Unit)**
- Key Concepts:
  - Sliding window technique
  - Time series preprocessing
  - Sequence learning

---

### 🔹 5. GAN - Image Generation (CIFAR-10)
- Built using **PyTorch**
- Consists of:
  - Generator (creates images)
  - Discriminator (detects fake images)
- Demonstrates adversarial training

---

### 🔹 6. Autoencoder - MNIST Image Reconstruction 🧠
- Built using **TensorFlow / Keras**
- Learns to **compress and reconstruct images**
- Dataset: MNIST (28×28 grayscale digits)
- Consists of:
  - Encoder → Compresses image into **latent space**
  - Decoder → Reconstructs image from compressed representation
  - Latent Dimension: 64
  - Loss Function: Mean Squared Error (MSE)

---

### 🔹 7. LSTM - Time Series Forecasting (Milk Production) 📈
- Built using **TensorFlow / Keras**
- Predicts **monthly milk production** based on past data
- Dataset: Monthly Milk Production (Time Series Data)
- Consists of :
  - LSTM (Long Short-Term Memory) → captures **long-term dependencies**
  - Sliding Window Technique → uses past 12 months to predict next value
  - Time Series Forecasting
  - Data Normalization using MinMaxScaler

---

## ⚙️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/Krishna28072000/Deep-Learning-Modules-.git
cd Deep-Learning-Modules-
```

Installations : pip install numpy pandas matplotlib scikit-learn tensorflow torch torchvision
