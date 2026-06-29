"""
Deep Learning with Python, 3rd Edition — Chapter 4
Example 2: Classifying newswires (multiclass classification) on the Reuters dataset.

Run:  python example_02_reuters_multiclass_classification.py

By default plots are saved to PNG files so this runs headless in a Codespace.
Pass --show to open plot windows instead.
"""

import argparse
import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import matplotlib

import copy
import numpy as np
import keras
from keras import layers
from keras.datasets import reuters
from keras.utils import to_categorical


def multi_hot_encode(sequences, num_classes):
    results = np.zeros((len(sequences), num_classes), dtype="float32")
    for i, sequence in enumerate(sequences):
        results[i][sequence] = 1.0
    return results


def build_model(num_classes=46):
    model = keras.Sequential(
        [
            layers.Dense(64, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def main(show=False):
    import matplotlib.pyplot as plt

    # ---- The Reuters dataset ----
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(
        num_words=10000
    )
    print("Training examples:", len(train_data))
    print("Test examples:", len(test_data))

    # Decode an example newswire back to words (illustrative only).
    word_index = reuters.get_word_index()
    reverse_word_index = {value: key for (key, value) in word_index.items()}
    decoded_newswire = " ".join(
        reverse_word_index.get(i - 3, "?") for i in train_data[10]
    )
    print("Sample label:", train_labels[10])
    print("Decoded newswire (first 100 chars):", decoded_newswire[:100])

    # ---- Preparing the data ----
    x_train = multi_hot_encode(train_data, num_classes=10000)
    x_test = multi_hot_encode(test_data, num_classes=10000)

    # One-hot encode the labels. (to_categorical does the same thing as the
    # hand-written one_hot_encode in the book.)
    y_train = to_categorical(train_labels)
    y_test = to_categorical(test_labels)

    # ---- Build and compile the model, including a top-3 accuracy metric ----
    top_3_accuracy = keras.metrics.TopKCategoricalAccuracy(
        k=3, name="top_3_accuracy"
    )
    model = build_model()
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy", top_3_accuracy],
    )

    # ---- Validating your approach (hold-out validation) ----
    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]
    y_val = y_train[:1000]
    partial_y_train = y_train[1000:]

    history = model.fit(
        partial_x_train,
        partial_y_train,
        epochs=20,
        batch_size=512,
        validation_data=(x_val, y_val),
    )

    # Plot loss.
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, "r--", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.xticks(epochs)
    plt.ylabel("Loss")
    plt.legend()
    _output(plt, "reuters_loss.png", show)

    # Plot accuracy.
    plt.clf()
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    plt.plot(epochs, acc, "r--", label="Training accuracy")
    plt.plot(epochs, val_acc, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.xticks(epochs)
    plt.ylabel("Accuracy")
    plt.legend()
    _output(plt, "reuters_accuracy.png", show)

    # Plot top-3 accuracy.
    plt.clf()
    acc = history.history["top_3_accuracy"]
    val_acc = history.history["val_top_3_accuracy"]
    plt.plot(epochs, acc, "r--", label="Training top-3 accuracy")
    plt.plot(epochs, val_acc, "b", label="Validation top-3 accuracy")
    plt.title("Training and validation top-3 accuracy")
    plt.xlabel("Epochs")
    plt.xticks(epochs)
    plt.ylabel("Top-3 accuracy")
    plt.legend()
    _output(plt, "reuters_top3_accuracy.png", show)

    # ---- Retrain from scratch for 9 epochs and evaluate on the test set. ----
    model = build_model()
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(x_train, y_train, epochs=9, batch_size=512)
    results = model.evaluate(x_test, y_test)
    print("Test loss / accuracy:", results)

    # Compare against a random baseline.
    test_labels_copy = copy.copy(test_labels)
    np.random.shuffle(test_labels_copy)
    random_baseline = np.mean(np.array(test_labels) == test_labels_copy)
    print("Random-classifier baseline accuracy:", random_baseline)

    # ---- Generating predictions on new data ----
    predictions = model.predict(x_test)
    print("Prediction vector shape:", predictions[0].shape)
    print("Prediction probabilities sum to:", np.sum(predictions[0]))
    print("Predicted class for first example:", np.argmax(predictions[0]))

    # ---- Alternative: integer labels with sparse_categorical_crossentropy ----
    # (Same model, no one-hot encoding needed for the labels.)
    y_train_int = train_labels
    y_test_int = test_labels
    sparse_model = build_model()
    sparse_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    sparse_model.fit(x_train, y_train_int, epochs=9, batch_size=512)
    sparse_results = sparse_model.evaluate(x_test, y_test_int)
    print("Sparse-label model test loss / accuracy:", sparse_results)


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
        matplotlib.use("Agg")
    main(show=args.show)
