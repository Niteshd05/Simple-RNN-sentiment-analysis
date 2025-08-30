Sentiment Analysis on IMDB Movie Reviews 🎬📊

This project implements a Sentiment Analysis model using a Simple Recurrent Neural Network (RNN) to classify IMDB movie reviews as positive or negative.

📌 Project Overview

Dataset: IMDB Movie Reviews (50,000 labeled reviews)

Task: Binary Sentiment Classification

Model: Simple RNN built with TensorFlow/Keras

Output:

1 → Positive review

0 → Negative review

⚙️ Features

✅ Preprocessing of IMDB reviews (tokenization, padding)
✅ Embedding layer for word representation
✅ Simple RNN architecture for sequential data learning
✅ Model training with validation
✅ Save & load trained model (.h5 format)
✅ Prediction on custom reviews

🏗️ Model Architecture
Embedding Layer → SimpleRNN Layer → Dense Layer (ReLU) → Output Layer (Sigmoid)

🚀 Installation

Clone the repository:

git clone https://github.com/your-username/sentiment-rnn-imdb.git
cd sentiment-rnn-imdb


Install dependencies:

pip install -r requirements.txt


Run the training script:

python train.py


Run the inference / demo:

python main.py

📂 Project Structure
📦 sentiment-rnn-imdb
 ┣ 📜 train.py          # Training script
 ┣ 📜 main.py           # Prediction / demo script
 ┣ 📜 simple_rnn_imdb.h5 # Saved trained model
 ┣ 📜 requirements.txt  # Dependencies
 ┣ 📜 README.md         # Project documentation

🔍 Example Usage
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# Load model
model = load_model("simple_rnn_imdb.h5")

# Example review (already preprocessed into tokens & padded)
sample = x_test[0].reshape(1, -1)
prediction = model.predict(sample)

print("Positive" if prediction[0][0] > 0.5 else "Negative")

📊 Results

Accuracy on Test Set: ~85% (may vary depending on training configuration)

Loss and accuracy curves available in training logs.

📌 Future Improvements

Replace SimpleRNN with LSTM/GRU for better handling of long sequences.

Use pretrained embeddings (e.g., GloVe, Word2Vec).

Add attention mechanism for interpretability.

Deploy as a Flask / Streamlit web app for interactive predictions.
