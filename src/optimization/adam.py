# optimization/adam.py

import numpy as np
import time
from typing import Tuple, List
from utils.helpers import logistic_loss

def adam_optimizer(X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, learning_rate: float = 0.001, n_iter: int = 1000, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8, tolerance: float = 1e-6) -> Tuple[np.ndarray, List[float], List[float], float]:
    """
    Implémente l'algorithme Adam (Adaptive Moment Estimation).

    Args:
        X (np.ndarray): Matrice des caractéristiques d'entraînement.
        y (np.ndarray): Vecteur des labels d'entraînement.
        X_val (np.ndarray): Matrice des caractéristiques de validation.
        y_val (np.ndarray): Vecteur des labels de validation.
        learning_rate (float): Taux d'apprentissage. Par défaut à 0.001.
        n_iter (int): Nombre maximal d'itérations. Par défaut à 1000.
        beta1 (float): Paramètre de décroissance pour le premier moment. Par défaut à 0.9.
        beta2 (float): Paramètre de décroissance pour le deuxième moment. Par défaut à 0.999.
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
    m = np.zeros_like(weights)
    v = np.zeros_like(weights)
    train_losses = []
    val_losses = []

    start_time = time.time()

    for t in range(1, n_iter + 1):
        predictions = 1 / (1 + np.exp(-X @ weights))
        grad = -X.T @ (y - predictions) / len(y)

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        weights -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        if np.linalg.norm(grad) < tolerance:
            break

        train_loss = logistic_loss(weights, X, y)
        val_loss = logistic_loss(weights, X_val, y_val)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    end_time = time.time()
    convergence_time = end_time - start_time

    return weights, train_losses, val_losses, convergence_time