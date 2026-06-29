"""
Deep Learning with Python, 3rd Edition — Chapter 4
Example 1: Classifying movie reviews (binary classification) on the IMDb dataset.

Run:  python example_01_imdb_binary_classification.py

By default this script saves plots to PNG files instead of opening windows,
so it works fine in a headless GitHub Codespace. Pass --show to open windows
if you have a display.
"""

import argparse
import os

# Pick a Keras backend. JAX is the book's default; "tensorflow" or "torch"
# also work. Must be set before importing keras.
os.environ.setdefault("KERAS_BACKEND", "jax")

import matplotlib

import numpy as np
import keras
from keras import layers
from keras.datasets import imdb


def multi_hot_encode(sequences, num_classes):
    results = np.zeros((len(sequences), num_classes), dtype="float32")
    for i, sequence in enumerate(sequences):
        results[i][sequence] = 1.0
    return results


def build_model():
    model = keras.Sequential(
        [
            layers.Dense(16, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main(show=False):
    import matplotlib.pyplot as plt

    # ---- The IMDb dataset ----
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(
        num_words=10000
    )

    print("First training sequence (first 10 tokens):", train_data[0][:10])
    print("First training label:", train_labels[0])
    print("Largest word index used:", max(max(seq) for seq in train_data))

    # Decode a review back to words (just to show how it's done).
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for (key, value) in word_index.items()}
    decoded_review = " ".join(
        reverse_word_index.get(i - 3, "?") for i in train_data[0]
    )
    print("Decoded review (first 100 chars):", decoded_review[:100])

    # ---- Preparing the data ----
    x_train = multi_hot_encode(train_data, num_classes=10000)
    x_test = multi_hot_encode(test_data, num_classes=10000)
    y_train = train_labels.astype("float32")
    y_test = test_labels.astype("float32")

    # ---- Validating your approach (hold-out validation) ----
    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]
    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]

    model = build_model()
    history = model.fit(
        partial_x_train,
        partial_y_train,
        epochs=20,
        batch_size=512,
        validation_data=(x_val, y_val),
    )

    history_dict = history.history

    # Plot training/validation loss.
    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, "r--", label="Training loss")
    plt.plot(epochs, val_loss_values, "b", label="Validation loss")
    plt.title("[IMDB] Training and validation loss")
    plt.xlabel("Epochs")
    plt.xticks(epochs)
    plt.ylabel("Loss")
    plt.legend()
    _output(plt, "imdb_loss.png", show)

    # Plot training/validation accuracy.
    plt.clf()
    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]
    plt.plot(epochs, acc, "r--", label="Training acc")
    plt.plot(epochs, val_acc, "b", label="Validation acc")
    plt.title("[IMDB] Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.xticks(epochs)
    plt.ylabel("Accuracy")
    plt.legend()
    _output(plt, "imdb_accuracy.png", show)

    # ---- Retrain from scratch for only 4 epochs (the validation curves
    # show the model starts overfitting after ~4 epochs) and evaluate. ----
    model = build_model()
    model.fit(x_train, y_train, epochs=4, batch_size=512)
    results = model.evaluate(x_test, y_test)
    print("Test loss / accuracy:", results)

    # ---- Generate predictions on new data ----
    preds = model.predict(x_test)
    print("First 10 prediction probabilities:", preds[:10].ravel())


def _output(plt, filename, show):
    if show:
        plt.show()
    else:
        plt.savefig(filename, dpi=120, bbox_inches="tight")
        print(f"Saved plot to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open plot windows instead of saving PNG files.",
    )
    args = parser.parse_args()
    if not args.show:
        matplotlib.use("Agg")  # headless backend for Codespaces
    main(show=args.show)
