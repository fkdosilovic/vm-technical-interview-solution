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

    def fit(self, X: np.ndarray, y: np.ndarray, callbacks: dict = None) -> None:
        """Train the k-nearest neighbors model.

        Training the model consists of fitting the nearest neighbors model to
        the input data and storning the data and labels.

        Parameters
        ----------
        X : np.ndarray
            The input data arranged in a matrix. Each row represents an example
            and each column represents a feature.
        y : np.ndarray
            The labels of the input data. The labels must be integers in the
            range [0, n_classes).
        callbacks : dict, optional
            Ignored.

        Returns
        -------
        None
        """
        self.nn.fit(X)

        self.X = X
        self.y = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the indices of the k nearest neighbors.

        Parameters
        ----------
        X : np.ndarray
            The input data arranged in a matrix. Each row represents an example
            and each column represents a feature. The number of columns must
            match the number of features used to fit the model.

        Returns
        -------
        np.ndarray
            An array of shape (n_examples,) containing the predicted class for
            each example.
        """
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

        return np.apply_along_axis(
            _most_common_label_in_row, axis=1, arr=labels
        ).astype(np.int64)


class LogisticRegressionClassifier(BaseClassifier):
    """
    Implements a logistic regression model.

    Logistic regression, also called softmax regression, is a linear model for
    classification. It is a generalization of the linear regression model for
    classification problems with more than two classes. It is a parametric
    model that learns a weight matrix W and a bias vector b. The model uses
    a standard formulation of the softmax function to compute the probabilities
    of each class for each example as well as a standard cross-entropy loss
    function to compute the loss and the gradients.
    """

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

    def fit(self, X: np.ndarray, y: np.ndarray, callbacks: dict = None) -> None:
        """Train the logistic regression model.

        The models is trained using a mini-batch gradient descent.

        Parameters
        ----------
        X : np.ndarray
            The input data arranged in a matrix. Each row represents an example
            and each column represents a feature.
        y : np.ndarray
            The labels of the input data. The labels must be integers in the
            range [0, n_classes).

        Returns
        ------
        None
        """
        if len(X) != len(y):
            raise ValueError("The number of examples must match the number of labels.")

        n_classes = len(np.unique(y))
        self.W = np.random.randn(n_classes, X.shape[1])
        self.b = np.random.randn(n_classes)

        Y = np.eye(n_classes)[y]  # Get one-hot encoded labels.

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

            # TODO: Add callbacks for early stopping and logging loss and metrics.

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return the probabilities of the classes for each example.

        Parameters
        ----------
        X : np.ndarray
            The input data arranged in a matrix. Each row represents an example
            and each column represents a feature. The number of columns must
            match the number of features used to fit the model.

        Returns
        -------
        np.ndarray
            A matrix of shape (n_examples, n_classes) containing the
            probabilities of each class for each example.
        """
        if self.W is None or self.b is None:
            raise ValueError(
                "The model has not been fitted yet. Call the fit method first."
            )

        if X.ndim == 1:
            X = X.reshape(1, -1)

        logits = np.matmul(X, self.W.T) + self.b
        return np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the class for each example.

        Parameters
        ----------
        X : np.ndarray
            The input data arranged in a matrix. Each row represents an example
            and each column represents a feature. The number of columns must
            match the number of features used to fit the model.

        Returns
        -------
        np.ndarray
            An array of shape (n_examples,) containing the predicted class for
            each example.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1).astype(np.int64)


class NadarayaWatsonClassifier(BaseClassifier):
    """Implements a Nadaraya-Watson classifier.

    The Nadaraya-Watson classifier is a non-parametric classifier that
    belongs to the class of kernely density estimators. It is based on the idea
    of computing the weighted average of the labels of the k nearest neighbors
    of a query point. The weights are computed using a kernel function.

    The kernel function used in this implementation is the Gaussian kernel.

    More details can be found in the following link: https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-pooling.html
    """

    def __init__(self, nearest_neighbors: BaseNearestNeighbors):
        self.nn = nearest_neighbors

        self.X = None
        self.y = None

    def fit(self, X: np.ndarray, y: np.ndarray, callbacks: dict = None) -> None:
        """Train the Nadaraya-Watson classifier.

        Training the classifier consists of fitting the nearest neighbors model
        to the input data and storing the data and labels.

        Parameters
        ----------
        X : np.ndarray
            The input data arranged in a matrix. Each row represents an example
            and each column represents a feature.
        y : np.ndarray
            The labels of the input data. The labels must be integers in the
            range [0, n_classes).
        callbacks : dict, optional
            Ignored.
        """
        self.nn.fit(X)
        self.X = X
        self.y = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the class for each example.

        Parameters
        ----------
        X : np.ndarray
            The input data arranged in a matrix. Each row represents an example
            and each column represents a feature. The number of columns must
            match the number of features used to fit the model.

        Returns
        -------
        np.ndarray
            An array of shape (n_examples,) containing the predicted class for
            each example.
        """
        # For each query point get the indices of the k nearest neighbors.
        indices = self.nn.predict(X)

        neighborhood = self.X[indices]
        # For each query point get the labels of the k nearest neighbors.
        labels = self.y[indices]
        labels_oh = np.eye(len(np.unique(self.y)))[labels]

        # Goal: compute the weighted average of the labels of the k nearest neighbors.

        # First compute the pairwise distances between each query point and its neighbors.
        distances = np.linalg.norm(X[:, np.newaxis] - neighborhood, ord=2, axis=2)

        # Then compute the weights for each neighbor.
        weights = np.exp(-0.5 * distances**2)

        # Normalize the weights.
        weights_norm = weights / np.sum(weights, axis=1, keepdims=True)

        # Finally compute the weighted average of the labels.
        weights_norm = weights_norm[:, np.newaxis, :]
        predictions = np.matmul(weights_norm, labels_oh).squeeze()

        # Return the class with the highest score.
        return np.argmax(predictions, axis=1).astype(np.int64)
