import numpy as np

from abc import ABC, abstractmethod
from miniml.nearestneighbors import BaseNearestNeighbors


def _most_common_label_in_row(row):
    counts = np.bincount(row)
    return np.argmax(counts)


class BaseClassifier(ABC):
    """Base class for all classifiers."""

    def __init__(self) -> None:
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass


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
