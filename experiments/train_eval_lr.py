"""The following script is used to train the logistic regression model and evaluate it on the test set."""

import numpy as np

from miniml import load_mnist
from miniml import LogisticRegressionClassifier
from miniml import (
    compute_accuracy,
    precision_recall_fscore_support,
)

from tqdm import tqdm


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", type=str, help="Path to the MNIST dataset.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--n-samples", type=int, default=30000)

    return parser.parse_args()


def predict_batches(model, test_images, batch_size=256):
    n_batches = test_images.shape[0] // batch_size
    if test_images.shape[0] % batch_size != 0:
        n_batches += 1

    y_pred = np.empty(test_images.shape[0])
    for i in tqdm(range(n_batches)):
        start = i * batch_size
        end = start + batch_size
        y_pred[start:end] = model.predict(test_images[start:end])
    return y_pred


def main(args):
    # Load the dataset.
    X_train, y_train = load_mnist(args.dataset, split="train")
    X_test, y_test = load_mnist(args.dataset, split="test")

    # Sample training examples for faster training.
    np.random.seed(args.seed)
    indices = np.random.permutation(X_train.shape[0])[: args.n_samples]
    X_train = X_train[indices]
    y_train = y_train[indices]

    # Normalize the data.
    X_train = X_train.astype(np.float32) / 255
    X_test = X_test.astype(np.float32) / 255

    # Create the model.
    model = LogisticRegressionClassifier(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        validation_size=args.validation_size,
    )
    model.fit(X_train, y_train)

    # Evaluate the model.
    y_pred = predict_batches(model, X_test)

    y_test = y_test.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    accuracy = compute_accuracy(y_test, y_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred)

    print("Test set performance:\n")
    print(f"Accuracy:   {accuracy:10.4f}")
    print(f"Precision:  {np.mean(precision):10.4f}")
    print(f"Recall:     {np.mean(recall):10.4f}")
    print(f"F-score:    {np.mean(fscore):10.4f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
