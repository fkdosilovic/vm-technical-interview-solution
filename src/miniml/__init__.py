"""Minimal machine learning library built for the purposes of technical interview."""

from .classification import KNearestNeighborClassifier
from .classification import LogisticRegressionClassifier
from .classification import NadarayaWatsonClassifier

from .datasets import load_mnist

from .evaluation import (
    compute_accuracy,
    compute_confusion_matrix,
    precision_recall_fscore_support,
)

from .base import BaseNearestNeighbors
from .neighbors import NaiveNearestNeighbors
