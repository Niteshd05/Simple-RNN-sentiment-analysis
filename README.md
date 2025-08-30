# 🎬 Sentiment Analysis on IMDB Reviews using Simple RNN  

> **A Simple RNN-based deep learning model that classifies IMDB movie reviews into positive or negative sentiments.**  

---

## 📌 Overview  
This project implements **Sentiment Analysis** on the **IMDB dataset** (50,000 movie reviews) using a **Simple Recurrent Neural Network (RNN)** built with TensorFlow/Keras.  
The model learns to classify a review as **positive** or **negative** by analyzing its sequence of words.  

---

## ⚙️ Features  
- 🔹 Preprocessing of raw IMDB reviews (tokenization, word-index mapping, padding)  
- 🔹 Embedding layer for word vector representation  
- 🔹 Simple RNN architecture to capture sequential dependencies  
- 🔹 Model training with validation support  
- 🔹 Save & load trained model (`.h5` format)  
- 🔹 Predict sentiment for custom reviews  

---

## 🏗️ Model Architecture  
```text
Input (Padded Sequences)
        ↓
Embedding Layer
        ↓
SimpleRNN Layer (128 units)
        ↓
Dense Layer (ReLU)
        ↓
Output Layer (Sigmoid)
