import numpy as np
from abc import ABC, abstractmethod


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
