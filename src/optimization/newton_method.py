# optimization/newton_method.py

import numpy as np
import time
from typing import Tuple, List
from ..utils.helpers import logistic_loss

def newton_method(X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, n_iter: int = 100, tolerance: float = 1e-6) -> Tuple[np.ndarray, List[float], List[float], float]:
    """
    Implémente la méthode de Newton pour la régression logistique.

    Args:
        X (np.ndarray): Matrice des caractéristiques d'entraînement.
        y (np.ndarray): Vecteur des labels d'entraînement.
        X_val (np.ndarray): Matrice des caractéristiques de validation.
        y_val (np.ndarray): Vecteur des labels de validation.
        n_iter (int): Nombre maximal d'itérations. Par défaut à 100.
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

        # Calculer la Hessienne
        S = np.diag(predictions * (1 - predictions))
        H = X.T @ S @ X / len(y)

        try:
            weights -= np.linalg.solve(H, grad)  # Mise à jour par la méthode de Newton
        except np.linalg.LinAlgError:
            print("Erreur dans l'inversion de la Hessienne, arrêt de l'optimisation.")
            break

        if np.linalg.norm(grad) < tolerance:
            break

        train_loss = logistic_loss(weights, X, y)
        val_loss = logistic_loss(weights, X_val, y_val)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    end_time = time.time()
    convergence_time = end_time - start_time

    return weights, train_losses, val_losses, convergence_time


def quasi_newton_method(X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, n_iter: int = 100, tolerance: float = 1e-6) -> Tuple[np.ndarray, List[float], List[float], float]:
    """
    Implémente la méthode de quasi-Newton (BFGS) pour la régression logistique.

    Args:
        X (np.ndarray): Matrice des caractéristiques d'entraînement.
        y (np.ndarray): Vecteur des labels d'entraînement.
        X_val (np.ndarray): Matrice des caractéristiques de validation.
        y_val (np.ndarray): Vecteur des labels de validation.
        n_iter (int): Nombre maximal d'itérations. Par défaut à 100.
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

    # Matrice Hessienne approximative initiale
    H = np.eye(X.shape[1])
    start_time = time.time()

    for _ in range(n_iter):
        predictions = 1 / (1 + np.exp(-X @ weights))
        grad = -X.T @ (y - predictions) / len(y)

        # Mettre à jour les poids
        step = -np.linalg.inv(H) @ grad
        weights += step

        # Vérifier la convergence
        if np.linalg.norm(grad) < tolerance:
            break

        # Calculer la nouvelle matrice Hessienne approximative
        s = step
        y_diff = grad - (-X.T @ (y - predictions) / len(y))
        H = H + (s[:, np.newaxis] @ s[np.newaxis, :] / (s @ y_diff)) - (H @ (y_diff[:, np.newaxis] @ y_diff[np.newaxis, :]) @ H) / (y_diff @ H @ y_diff)

        # Calculer et enregistrer la perte
        train_loss = logistic_loss(weights, X, y)
        val_loss = logistic_loss(weights, X_val, y_val)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    end_time = time.time()
    convergence_time = end_time - start_time

    return weights, train_losses, val_losses, convergence_time