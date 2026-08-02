import json
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


MODEL_PATH = "imdb_results/best_model.keras"
TOKENIZER_PATH = "imdb_results/tokenizer.json"

MAX_LENGTH = 200


# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load tokenizer
with open(TOKENIZER_PATH, "r", encoding="utf-8") as file:
    tokenizer_data = json.load(file)

from tensorflow.keras.preprocessing.text import tokenizer_from_json

tokenizer = tokenizer_from_json(json.dumps(tokenizer_data))


def predict_sentiment(review):
    sequence = tokenizer.texts_to_sequences([review])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    prediction = model.predict(padded)[0][0]

    if prediction >= 0.5:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    confidence = prediction if prediction >= 0.5 else 1 - prediction

    return sentiment, confidence


print("IMDB Sentiment Analyzer")
print("Type 'quit' to exit\n")

while True:
    review = input("> ")

    if review.lower() in ["quit", "exit", "q"]:
        break

    sentiment, confidence = predict_sentiment(review)

    print(f"Prediction: {sentiment}")
    print(f"Confidence: {confidence:.2%}\n")