import numpy as np
from .base import BaseClassifier, BaseNearestNeighbors


def _most_common_label_in_row(row):
    counts = np.bincount(row)
    return np.argmax(counts)


class KNearestNeighborClassifier(BaseClassifier):
    """Implements a classifier based on the k-nearest neighbors algorithm."""

    def __init__(self, nearest_neighbors: BaseNearestNeighbors) -> None:
        super().__init__()
        self.nn = nearest_neighbors

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.nn.fit(X)

        self.X = X
        self.y = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the indices of the k nearest neighbors."""
        if self.X is None:
            raise ValueError(
                "The model has not been fitted yet. Call the fit method first."
            )

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # For each query point get the indices of the k nearest neighbors.
        indices = self.nn.predict(X)

        # For each query point get the labels of the k nearest neighbors.
        labels = self.y[indices]

        return np.apply_along_axis(_most_common_label_in_row, axis=1, arr=labels)


class LogisticRegressionClassifier(BaseClassifier):
    """Implements a logistic regression model."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        weight_decay: float = 0.001,
        batch_size: int = 32,
        n_epochs: int = 50,
    ):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.W = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_classes = len(np.unique(y))
        self.W = np.random.randn(n_classes, X.shape[1])
        self.b = np.random.randn(n_classes)

        Y = np.eye(n_classes)[y]

        for epoch in range(self.n_epochs):
            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X[i : i + self.batch_size]
                Y_batch = Y[i : i + self.batch_size]

                # Compute the gradients.
                P = self.predict_proba(X_batch)
                G = P - Y_batch
                dW = np.matmul(G.T, X_batch) / X_batch.shape[0]
                db = np.mean(G, axis=0)

                # Update the parameters.
                self.W -= (1 - self.weight_decay * self.learning_rate) * dW
                self.b -= self.learning_rate * db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return the probabilities of the classes for each example."""
        if self.W is None or self.b is None:
            raise ValueError(
                "The model has not been fitted yet. Call the fit method first."
            )

        if X.ndim == 1:
            X = X.reshape(1, -1)

        logits = np.matmul(X, self.W.T) + self.b
        return np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the predicted class for each example."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
