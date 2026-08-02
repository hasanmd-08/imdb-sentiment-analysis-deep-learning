# IMDB Deep Learning Sentiment Analysis

This project trains a TensorFlow/Keras model to classify a movie review as **good (positive)** or **bad (negative)**. It uses tokenization, fixed-length sequence padding, a learned word embedding, and a bidirectional LSTM.

## Install dependencies

```powershell
py -m pip install -r requirements.txt
```

## Train and predict

```powershell
py .\train_imdb_sentiment.py --dataset "C:\Users\fksha\Downloads\IMDB Dataset.csv" --output-dir .\imdb_results --interactive
```

Use this smaller run to verify everything quickly (10,000 reviews and up to three epochs):

```powershell
py .\train_imdb_sentiment.py --dataset "C:\Users\fksha\Downloads\IMDB Dataset.csv" --output-dir .\imdb_results_quick --quick
```

## Files produced

- `01_data_exploration.png` - sentiment balance and review-length distribution.
- `02_training_curves.png` - training and validation loss/accuracy using `tight_layout`.
- `03_confusion_matrix.png` - confusion matrix on the untouched test split.
- `best_model.keras` and `final_model.keras` - trained models.
- `tokenizer.json` and `model_metadata.json` - preprocessing settings for reuse.
- `test_predictions.csv` - every held-out prediction and its positive probability.

With `--interactive`, enter your own review after training. The program reports `GOOD (positive)` or `BAD (negative)` plus its confidence.
