"""Train an embedding + bidirectional LSTM on the Kaggle IMDB review dataset.

Creates exploratory charts, training curves, a confusion matrix, and reusable
model files. The CSV must contain `review` and `sentiment` columns.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

SEED = 42
DEFAULT_DATASET = Path(r"C:\Users\fksha\Downloads\IMDB Dataset.csv")


def clean_review(review: str) -> str:
    """Replace HTML breaks and excess whitespace without removing normal words."""
    review = re.sub(r"<br\s*/?>", " ", str(review), flags=re.I)
    return re.sub(r"\s+", " ", review).strip()


def save_exploration(frame: pd.DataFrame, destination: Path) -> None:
    """Show class balance and review length, which help assess dataset quality."""
    data = frame.copy()
    data["word_count"] = data.review.str.split().str.len()
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.countplot(data=data, x="sentiment", hue="sentiment", legend=False,
                  palette={"negative": "#d1495b", "positive": "#00798c"}, ax=axes[0])
    axes[0].set(title="Sentiment distribution", xlabel="Sentiment", ylabel="Reviews")
    sns.histplot(data=data, x="word_count", hue="sentiment", bins=50, element="step",
                 stat="density", common_norm=False, palette="Set2", ax=axes[1])
    axes[1].set_xlim(0, data.word_count.quantile(.99))
    axes[1].set(title="Review length (up to 99th percentile)", xlabel="Words per review", ylabel="Density")
    fig.tight_layout()
    fig.savefig(destination / "01_data_exploration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_model(vocab_size: int, max_length: int, embedding_dim: int) -> tf.keras.Model:
    """Use an embedding layer to learn semantic word vectors from the reviews."""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        Bidirectional(LSTM(64, dropout=.25)),
        Dense(64, activation="relu"),
        Dropout(.35),
        Dense(1, activation="sigmoid"),  # Probability of a positive review.
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def save_training_curves(history: tf.keras.callbacks.History, destination: Path) -> None:
    """Save loss and accuracy curves with a compact, readable tight layout."""
    h = history.history
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for key, title, ylabel, axis in [
        ("accuracy", "Model accuracy", "Accuracy", axes[0]),
        ("loss", "Model loss", "Binary cross-entropy", axes[1]),
    ]:
        axis.plot(h[key], label="Training", color="#00798c", marker="o")
        axis.plot(h[f"val_{key}"], label="Validation", color="#f4a261", marker="o")
        axis.set(title=title, xlabel="Epoch", ylabel=ylabel)
        axis.legend()
    fig.tight_layout()
    fig.savefig(destination / "02_training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_confusion(y_true: np.ndarray, y_pred: np.ndarray, destination: Path) -> None:
    """Save the held-out test performance as an easy-to-read confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=["Negative", "Positive"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Test-set confusion matrix")
    fig.tight_layout()
    fig.savefig(destination / "03_confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def predict_review(model: tf.keras.Model, tokenizer: Tokenizer, text: str, max_length: int) -> tuple[str, float]:
    """Pad a new review exactly as training data was padded, then classify it."""
    sequence = tokenizer.texts_to_sequences([clean_review(text)])
    padded = pad_sequences(sequence, maxlen=max_length, padding="post", truncating="post")
    positive_probability = float(model.predict(padded, verbose=0)[0][0])
    return ("GOOD (positive)" if positive_probability >= .5 else "BAD (negative)", positive_probability)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an IMDB deep learning sentiment classifier.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path("imdb_results"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=20_000)
    parser.add_argument("--max-length", type=int, default=250)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--quick", action="store_true", help="Train on 10,000 samples for up to three epochs.")
    parser.add_argument("--interactive", action="store_true", help="Accept new reviews after training.")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}. Use --dataset to provide its location.")
    frame = pd.read_csv(args.dataset)
    if not {"review", "sentiment"}.issubset(frame.columns):
        raise ValueError("The CSV must contain review and sentiment columns.")
    frame = frame.dropna(subset=["review", "sentiment"]).copy()
    frame.review = frame.review.map(clean_review)
    frame.sentiment = frame.sentiment.str.lower().str.strip()
    frame = frame[frame.sentiment.isin(["positive", "negative"])].reset_index(drop=True)
    if args.quick:
        frame, _ = train_test_split(frame, train_size=10_000, stratify=frame.sentiment, random_state=SEED)
        args.epochs = min(args.epochs, 3)
    print(f"Loaded {len(frame):,} reviews\n{frame.sentiment.value_counts().to_string()}")
    save_exploration(frame, args.output_dir)

    labels = (frame.sentiment == "positive").astype("int32").to_numpy()
    train_text, test_text, y_train, y_test = train_test_split(
        frame.review.to_numpy(), labels, test_size=.20, stratify=labels, random_state=SEED)
    train_text, validation_text, y_train, y_validation = train_test_split(
        train_text, y_train, test_size=.15, stratify=y_train, random_state=SEED)
    # Fit only on train data to prevent information leakage from validation/test data.
    tokenizer = Tokenizer(num_words=args.vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_text)
    vectorize = lambda texts: pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=args.max_length,
                                             padding="post", truncating="post")
    x_train, x_validation, x_test = vectorize(train_text), vectorize(validation_text), vectorize(test_text)
    model = build_model(args.vocab_size, args.max_length, args.embedding_dim)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=.5, patience=1, min_lr=1e-5),
        ModelCheckpoint(args.output_dir / "best_model.keras", monitor="val_loss", save_best_only=True),
    ]
    history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=args.epochs,
                        batch_size=128, callbacks=callbacks, verbose=2)
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    probability = model.predict(x_test, verbose=0).ravel()
    prediction = (probability >= .5).astype("int32")
    save_training_curves(history, args.output_dir)
    save_confusion(y_test, prediction, args.output_dir)
    model.save(args.output_dir / "final_model.keras")
    (args.output_dir / "tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")
    metadata = {"vocab_size": args.vocab_size, "max_length": args.max_length, "embedding_dim": args.embedding_dim,
                "test_loss": float(test_loss), "test_accuracy": float(test_accuracy)}
    (args.output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame({"review": test_text, "actual": y_test, "predicted": prediction,
                  "positive_probability": probability}).to_csv(args.output_dir / "test_predictions.csv", index=False)
    print(f"\nTest loss: {test_loss:.4f} | Test accuracy: {test_accuracy:.2%}")
    print(classification_report(y_test, prediction, target_names=["negative", "positive"]))
    print(f"Outputs saved to: {args.output_dir.resolve()}")
    if args.interactive:
        print("\nEnter a movie review. Type 'quit' to stop.")
        while True:
            text = input("> ").strip()
            if text.lower() in {"quit", "exit", "q"}:
                break
            if text:
                label, confidence = predict_review(model, tokenizer, text, args.max_length)
                print(f"Prediction: {label}; positive probability: {confidence:.1%}")


if __name__ == "__main__":
    main()
