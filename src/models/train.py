# models/train.py

import numpy as np
from typing import Tuple
from ..utils.helpers import logistic_loss

def train_logistic_regression(X: np.ndarray, y: np.ndarray, learning_rate: float = 0.01, n_iter: int = 1000, tolerance: float = 1e-6) -> Tuple[np.ndarray, list]:
    """
    Entraîne un modèle de régression logistique en utilisant la descente de gradient.

    Args:
        X (np.ndarray): Matrice des caractéristiques.
        y (np.ndarray): Vecteur des labels.
        learning_rate (float): Taux d'apprentissage. Par défaut à 0.01.
        n_iter (int): Nombre maximal d'itérations. Par défaut à 1000.
        tolerance (float): Tolérance pour la convergence. Par défaut à 1e-6.

    Returns:
        Tuple[np.ndarray, list]:
            - weights: Vecteur de poids entraîné.
            - losses: Liste des pertes à chaque itération.
    """
    weights = np.random.randn(X.shape[1]) * 0.01
    losses = []

    for _ in range(n_iter):
        predictions = 1 / (1 + np.exp(-X @ weights))
        grad = -X.T @ (y - predictions) / len(y)
        new_weights = weights - learning_rate * grad

        if np.linalg.norm(new_weights - weights) < tolerance:
            break

        weights = new_weights
        loss = logistic_loss(weights, X, y)
        losses.append(loss)

    return weights, losses