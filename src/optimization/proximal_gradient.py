# optimization/proximal_gradient.py

import numpy as np
import time
from typing import Tuple, List
from ..utils.helpers import logistic_loss, soft_thresholding

def proximal_gradient_descent(X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, learning_rate: float = 0.01, n_iter: int = 1000, lambda_: float = 0.1, tolerance: float = 1e-6) -> Tuple[np.ndarray, List[float], List[float], float]:
    """
    Implémente le gradient proximal pour la régression logistique avec régularisation L1.

    Args:
        X (np.ndarray): Matrice des caractéristiques d'entraînement.
        y (np.ndarray): Vecteur des labels d'entraînement.
        X_val (np.ndarray): Matrice des caractéristiques de validation.
        y_val (np.ndarray): Vecteur des labels de validation.
        learning_rate (float): Taux d'apprentissage. Par défaut à 0.01.
        n_iter (int): Nombre maximal d'itérations. Par défaut à 1000.
        lambda_ (float): Paramètre de régularisation L1. Par défaut à 0.1.
        tolerance (float): Tolérance pour la convergence. Par défaut à 1e-6.

    Returns:
        Tuple[np.ndarray, List[float], List[float], float]:
            - weights: Vecteur de poids entraîné.
            - train_losses: Liste des pertes d'entraînement.
            - val_losses: Liste des pertes de validation.
            - convergence_time: Temps de convergence en secondes.
    """
    weights = np.random.randn(X.shape[1]) * 0.01
    train_losses = []
    val_losses = []

    start_time = time.time()

    for _ in range(n_iter):
        predictions = 1 / (1 + np.exp(-X @ weights))
        grad = -X.T @ (y - predictions) / len(y)
        new_weights = weights - learning_rate * grad

        # Appliquer la régularisation L1
        new_weights = np.array([soft_thresholding(w, learning_rate * lambda_) for w in new_weights])

        if np.linalg.norm(new_weights - weights) < tolerance:
            break

        weights = new_weights
        train_loss = logistic_loss(weights, X, y)
        val_loss = logistic_loss(weights, X_val, y_val)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    end_time = time.time()
    convergence_time = end_time - start_time

    return weights, train_losses, val_losses, convergence_time