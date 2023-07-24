import numpy as np

from abc import ABC, abstractmethod


class BaseNearestNeighbors(ABC):
    def __init__(self, k: int = 3):
        self.k = k

        if self.k < 1:
            raise ValueError("k must be greater than 0.")

        self.X = None

    @abstractmethod
    def fit(self, X: np.ndarray):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass


class NaiveNearestNeighbors(BaseNearestNeighbors):
    """
    Implementation of the naive nearest neighbors algorithm.

    The algorithm works by computing the distance between the query point(s) and
    all the points in the training set. The k nearest points are then selected
    and the indices of the k nearest points (from the training set) are returned.
    The algorithm is called "naive" because it does not use any data structure
    to speed up the search for the nearest neighbors.
    """

    def __init__(self, k: int = 3):
        """Initialize the NaiveNearestNeighbors model."""
        super().__init__(k)

    def fit(self, X: np.ndarray):
        """Train the model."""
        self.X = X

    def predict(self, X_query: np.ndarray) -> np.ndarray:
        """Return the indices of the k nearest neighbors."""
        if self.X is None:
            raise ValueError(
                "The model has not been fitted yet. Call the fit method first."
            )

        if X_query.ndim == 1:
            X_query = X_query.reshape(1, -1)

        if X_query.shape[1] != self.X.shape[1]:
            raise ValueError(
                "The number of features in query points must be equal to the number of features in the training set."
            )

        # TODO: Add support for different distance metrics.
        # Compute the distances between the query points and the training set.
        distances = np.linalg.norm(X_query[:, np.newaxis] - self.X, axis=2)

        # For each query point get the indices of the k nearest neighbors.
        return np.argsort(distances, axis=1)[:, : self.k]
