"""Minimal machine learning library built for the purposes of technical interview."""

from .classification import KNearestNeighborClassifier
from .classification import LogisticRegressionClassifier
from .classification import NadarayaWatsonClassifier

from .datasets import load_mnist

from .base import BaseNearestNeighbors
from .neighbors import NaiveNearestNeighbors
