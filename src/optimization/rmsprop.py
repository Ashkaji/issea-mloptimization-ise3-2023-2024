# optimization/rmsprop.py

import numpy as np
import time
from typing import Tuple, List
from utils.helpers import logistic_loss

def rmsprop_optimizer(X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, learning_rate: float = 0.001, n_iter: int = 1000, decay_rate: float = 0.9, epsilon: float = 1e-8, tolerance: float = 1e-6) -> Tuple[np.ndarray, List[float], List[float], float]:
    """
    Implémente l'algorithme RMSprop (Root Mean Square Propagation).

    Args:
        X (np.ndarray): Matrice des caractéristiques d'entraînement.
        y (np.ndarray): Vecteur des labels d'entraînement.
        X_val (np.ndarray): Matrice des caractéristiques de validation.
        y_val (np.ndarray): Vecteur des labels de validation.
        learning_rate (float): Taux d'apprentissage. Par défaut à 0.001.
        n_iter (int): Nombre maximal d'itérations. Par défaut à 1000.
        decay_rate (float): Taux de décroissance pour la moyenne des carrés des gradients. Par défaut à 0.9.
        epsilon (float): Terme de régularisation pour éviter la division par zéro. Par défaut à 1e-8.
        tolerance (float): Tolérance pour la convergence. Par défaut à 1e-6.

    Returns:
        Tuple[np.ndarray, List[float], List[float], float]:
            - weights: Vecteur de poids entraîné.
            - train_losses: Liste des pertes d'entraînement.
            - val_losses: Liste des pertes de validation.
            - convergence_time: Temps de convergence en secondes.
    """
    weights = np.random.randn(X.shape[1]) * 0.01
    avg_squared_grad = np.zeros_like(weights)
    train_losses = []
    val_losses = []

    start_time = time.time()

    for _ in range(n_iter):
        predictions = 1 / (1 + np.exp(-X @ weights))
        grad = -X.T @ (y - predictions) / len(y)

        avg_squared_grad = decay_rate * avg_squared_grad + (1 - decay_rate) * (grad ** 2)
        weights -= learning_rate * grad / (np.sqrt(avg_squared_grad) + epsilon)

        if np.linalg.norm(grad) < tolerance:
            break

        train_loss = logistic_loss(weights, X, y)
        val_loss = logistic_loss(weights, X_val, y_val)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    end_time = time.time()
    convergence_time = end_time - start_time

    return weights, train_losses, val_losses, convergence_time