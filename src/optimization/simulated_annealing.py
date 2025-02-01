# optimization/simulated_annealing.py

import numpy as np
import time
from typing import Tuple, List
from ..utils.helpers import logistic_loss

def simulated_annealing(X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, initial_temp: float = 1.0, final_temp: float = 1e-3, alpha: float = 0.9, n_iter: int = 1000) -> Tuple[np.ndarray, List[float], List[float], float]:
    """
    Implémente l'algorithme de recuit simulé pour l'optimisation.

    Args:
        X (np.ndarray): Matrice des caractéristiques d'entraînement.
        y (np.ndarray): Vecteur des labels d'entraînement.
        X_val (np.ndarray): Matrice des caractéristiques de validation.
        y_val (np.ndarray): Vecteur des labels de validation.
        initial_temp (float): Température initiale. Par défaut à 1.0.
        final_temp (float): Température finale. Par défaut à 1e-3.
        alpha (float): Facteur de refroidissement. Par défaut à 0.9.
        n_iter (int): Nombre maximal d'itérations. Par défaut à 1000.

    Returns:
        Tuple[np.ndarray, List[float], List[float], float]:
            - weights: Vecteur de poids entraîné.
            - train_losses: Liste des pertes d'entraînement.
            - val_losses: Liste des pertes de validation.
            - convergence_time: Temps de convergence en secondes.
    """
    weights = np.random.randn(X.shape[1]) * 0.01
    current_loss = logistic_loss(weights, X, y)
    train_losses = []
    val_losses = []

    temperature = initial_temp
    start_time = time.time()

    for _ in range(n_iter):
        new_weights = weights + np.random.normal(0, 0.1, size=weights.shape)
        new_loss = logistic_loss(new_weights, X, y)

        delta_loss = new_loss - current_loss

        if delta_loss < 0 or np.random.rand() < np.exp(-delta_loss / temperature):
            weights = new_weights
            current_loss = new_loss

        train_loss = logistic_loss(weights, X, y)
        val_loss = logistic_loss(weights, X_val, y_val)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        temperature *= alpha

        if temperature < final_temp:
            break

    end_time = time.time()
    convergence_time = end_time - start_time

    return weights, train_losses, val_losses, convergence_time