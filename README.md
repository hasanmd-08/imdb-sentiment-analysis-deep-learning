# 🎬 IMDB Movie Review Sentiment Analysis using BiLSTM

My first **Deep Learning Natural Language Processing (NLP)** project — building a **Bidirectional LSTM (BiLSTM)** model to classify movie reviews as **positive or negative** using the IMDB dataset.

## 📌 Overview

This project demonstrates an end-to-end Deep Learning workflow for a binary sentiment classification problem using text data.

The model learns sentiment patterns from movie reviews by converting text into numerical sequences using **tokenization and padding**, learning meaningful word representations through **embedding layers**, and performing classification using a **Bidirectional LSTM neural network**.

The project covers:

- Dataset exploration and visualization
- Text preprocessing
- Tokenization of movie reviews
- Sequence padding
- Word embeddings
- Building a Bidirectional LSTM model
- Model training and validation
- Model evaluation using classification metrics
- Confusion matrix analysis
- Real-time sentiment prediction on custom reviews

📝 **Full write-up:** [Read the Medium article](PASTE_YOUR_MEDIUM_ARTICLE_LINK_HERE)

---

## 🎯 Objective

- Build my first Deep Learning NLP classification model.
- Understand how neural networks process and learn from text data.
- Learn the complete workflow of an NLP project.
- Apply tokenization and padding techniques for sequence processing.
- Understand the importance of word embeddings in NLP.
- Build and evaluate a Bidirectional LSTM sentiment classifier.
- Create a system that predicts sentiment from new movie reviews.

---

## 🛠️ Tools & Technologies

- Python
- TensorFlow / Keras — Deep Learning framework
- NumPy — Numerical computations
- Pandas — Data manipulation
- Scikit-learn — Model evaluation
- Matplotlib — Data visualization
- Seaborn — Statistical visualization
- Google Colab / VS Code — Development environment

---

## 🧠 Deep Learning Architecture

The model follows this workflow:

```text
Movie Review Text
        |
        ↓
Tokenization
        |
        ↓
Sequence Padding
        |
        ↓
Embedding Layer
        |
        ↓
Bidirectional LSTM
        |
        ↓
Dense Layer
        |
        ↓
Sigmoid Activation
        |
        ↓
Positive / Negative Sentiment
```

---

## 📊 Key Findings

- Explored the IMDB dataset containing **50,000 movie reviews**.
- Processed text data into numerical sequences suitable for neural networks.
- Applied padding to maintain consistent input length.
- Built a Bidirectional LSTM model for sentiment classification.
- Achieved **87.30% test accuracy** on unseen movie reviews.
- Evaluated performance using precision, recall, F1-score, and confusion matrix.
- Created an interactive prediction system for analyzing custom reviews.

---

## 📁 Project Structure

```text
imdb_sentiment_project
│
├── train_imdb_sentiment.py       # Training pipeline
├── predict_sentiment.py          # Sentiment prediction script
├── requirements.txt              # Project dependencies
├── README.md                     # Project documentation
├── .gitignore                    # Ignored files
│
└── results
    ├── 01_data_exploration.png
    ├── 02_training_curves.png
    ├── 03_confusion_matrix.png
    └── test_predictions.csv
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/hasanmd-08/imdb-sentiment-analysis-deep-learning.git
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

Activate:

Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python train_imdb_sentiment.py
```

### 5. Predict sentiment on custom reviews

```bash
python predict_sentiment.py
```

---

## 🔮 Future Improvements

- Experiment with advanced NLP architectures such as Transformers.
- Compare BiLSTM performance with GRU and Transformer models.
- Use pretrained word embeddings such as GloVe or Word2Vec.
- Perform hyperparameter tuning for improved performance.
- Deploy the sentiment analysis model using Streamlit.

---

## 🔗 Connect

- 📝 **Medium:** https://medium.com/@hasan.huraira704
- 💼 **LinkedIn:** https://www.linkedin.com/in/hasan-mohamed-926230395

---

⭐ If you found this project useful, consider giving it a star!