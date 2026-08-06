# 🎬 IMDB Sentiment Analysis with Bidirectional LSTM

A binary sentiment classifier trained on 50,000 IMDB movie reviews using a Bidirectional LSTM built with TensorFlow/Keras.

---

## 📌 Overview

This project builds a deep learning model that classifies movie reviews as positive or negative. The model uses an embedding layer to learn word representations, followed by a Bidirectional LSTM to capture context from both directions in a review sequence. Training and inference are split into two separate scripts, designed to be run from the command line.

📝 **Full write-up:** [Read the Medium article](https://medium.com/@hasan.huraira704/cceda15fec17) 

---

## 🎯 Objective

To build and evaluate a text classification model that can determine whether a movie review expresses a positive or negative sentiment, using a Bidirectional LSTM architecture on the standard IMDB dataset.

---

## 🛠️ Tools

- **Python 3.10+**
- **TensorFlow / Keras** — model architecture, training, callbacks
- **scikit-learn** — train/test split, classification report, confusion matrix
- **pandas / numpy** — data handling and preprocessing
- **matplotlib / seaborn** — exploratory charts and training visualizations
- **VS Code** — development environment

---

## 📊 Key Findings

| Metric | Value |
|---|---|
| Test Accuracy | **87.30%** |
| Test Loss | 0.3106 |
| Precision (Negative) | 0.87 |
| Precision (Positive) | 0.88 |
| Recall (Negative) | 0.88 |
| Recall (Positive) | 0.87 |
| F1 Score (Both Classes) | 0.87 |
| Total Test Samples | 10,000 |

**Confusion Matrix:**
- True Negatives: 4,397 | False Positives: 603
- False Negatives: 667 | True Positives: 4,333

Training ran for 4 epochs before EarlyStopping triggered. Best weights were from epoch 1. A visible gap between training (~94.6%) and validation (~87%) accuracy indicates mild overfitting.

---

## 📁 Project Structure

```
imdb-sentiment-bilstm/
│
├── train_imdb_sentiment.py     # Full training pipeline: preprocessing, model, evaluation, output saving
├── predict_sentiment.py        # Inference script: loads saved model and tokenizer, accepts user input
│
├── imdb_results/               # Generated on training run (not committed)
│   ├── best_model.keras        # Best model weights by validation loss
│   ├── final_model.keras       # Model weights at end of training
│   ├── tokenizer.json          # Saved tokenizer for inference
│   ├── model_metadata.json     # Training config (vocab size, max_length, embedding_dim, test metrics)
│   ├── test_predictions.csv    # Review text, actual labels, predicted labels, probabilities
│   ├── 01_data_exploration.png
│   ├── 02_training_curves.png
│   └── 03_confusion_matrix.png
│
├── .gitignore
└── README.md
```

---

## 🚀 How to Run

**1. Clone the repo and install dependencies**
```bash
git clone https://github.com/hasanmd-08/imdb-sentiment-bilstm.git
cd imdb-sentiment-bilstm
pip install tensorflow scikit-learn pandas matplotlib seaborn
```

**2. Download the dataset**

Get the [IMDB Dataset from Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). The CSV must have `review` and `sentiment` columns.

**3. Train the model**
```bash
python train_imdb_sentiment.py --dataset path/to/IMDB\ Dataset.csv
```

Optional flags:
- `--epochs 10` — set max epochs (EarlyStopping may stop earlier)
- `--quick` — train on 10,000 samples for up to 3 epochs (faster iteration)
- `--interactive` — enter reviews manually after training to test live predictions

**4. Run inference**
```bash
python predict_sentiment.py
```

Type any movie review at the prompt. The script loads the saved model and tokenizer from `imdb_results/` and outputs a sentiment label with confidence score.

---

## 🔮 Future Improvements

- **Pre-trained embeddings** — initialize the embedding layer with GloVe or FastText weights instead of learning from scratch
- **Regularization** — add L2 penalties to dense layers to reduce the training/validation accuracy gap
- **Hyperparameter search** — systematic sweep over LSTM units, dropout rates, and learning rate
- **Stacked LSTMs** — experiment with two BiLSTM layers for deeper sequence modeling
- **Attention mechanism** — add attention to let the model weight which parts of a review matter most

---

## 🔗 Connect

- **Medium:** [Click Here](https://medium.com/@hasan.huraira704)
- **LinkedIn:** [Click Here](https://www.linkedin.com/in/hasan-mohamed-926230395)

---

⭐ If you found this project useful, consider giving it a star!
