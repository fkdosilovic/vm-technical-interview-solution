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


def hparam_optim_lr(X_train, y_train, X_valid, y_valid, args, **kwargs):
    best_lr, best_wd = None, None
    best_acc = 0.0

    for lr in kwargs.get("lr", [0.001, 0.005, 0.01, 0.05, 0.1]):
        for wd in kwargs.get(
            "weight_decay", [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25]
        ):
            # Create the model.
            model = LogisticRegressionClassifier(
                learning_rate=lr,
                weight_decay=wd,
                n_epochs=args.n_epochs,
                batch_size=args.batch_size,
                validation_size=args.validation_size,
            )
            model.fit(X_train, y_train)

            y_pred = predict_batches(model, X_valid)
            accuracy = compute_accuracy(y_valid, y_pred)
            print(f"lr={lr}, wd={wd}, accuracy={accuracy:.4f}")

            if accuracy > best_acc:
                best_acc = accuracy
                best_lr = lr
                best_wd = wd

    return {"lr": best_lr, "weight_decay": best_wd, "accuracy": best_acc}


def main(args):
    # Load the dataset.
    X_train, y_train = load_mnist(args.dataset, split="train")

    # Sample training examples for faster training.
    np.random.seed(args.seed)
    indices = np.random.permutation(X_train.shape[0])

    valid_size = int(args.validation_size * args.n_samples)
    X_valid = X_train[indices[args.n_samples : args.n_samples + valid_size]]
    y_valid = y_train[indices[args.n_samples : args.n_samples + valid_size]]
    X_train = X_train[indices[: args.n_samples]]
    y_train = y_train[indices[: args.n_samples]]

    # Normalize the data.
    X_train = X_train.astype(np.float32) / 255
    X_valid = X_valid.astype(np.float32) / 255

    print("Optimizing hyperparameters...")
    print(hparam_optim_lr(X_train, y_train, X_valid, y_valid, args))


if __name__ == "__main__":
    args = parse_args()
    main(args)
