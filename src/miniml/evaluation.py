"""Evaluation module implements standard evaluation metrics for machine learning."""

import numpy as np


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute the confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        The true labels.
    y_pred : np.ndarray
        The predicted labels.

    Returns
    -------
    np.ndarray
        The confusion matrix. The rows represent the true labels and the columns
        represent the predicted labels.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            "The shapes of the true labels and the predicted labels must be the same."
        )

    # Get the unique labels.
    labels = np.unique(y_true)

    # Initialize the confusion matrix.
    cm = np.zeros((len(labels), len(labels)), dtype=np.uint32)

    # For each label in the true labels and the predicted labels, increment the
    # corresponding cell in the confusion matrix.
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1

    return cm


def precision_recall_fscore_support(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Compute the precision, recall, f-score and support for each class.

    Parameters
    ----------
    y_true : np.ndarray
        The true labels.
    y_pred : np.ndarray
        The predicted labels.

    Returns
    -------
    tuple
        A tuple containing the precision, recall, f-score and support for each class.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            "The shapes of the true labels and the predicted labels must be the same."
        )

    cm = compute_confusion_matrix(y_true, y_pred)

    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    fscore = (2 * precision * recall) / (precision + recall)
    support = np.sum(cm, axis=1)

    return precision, recall, fscore, support


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the accuracy of the predictions."""
    if y_true.shape != y_pred.shape:
        raise ValueError(
            "The shapes of the true labels and the predicted labels must be the same."
        )

    return np.sum(y_true == y_pred) / len(y_true)
