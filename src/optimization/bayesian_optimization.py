# optimization/bayesian_optimization.py

import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import time
from typing import Tuple, List
from utils.helpers import logistic_loss

def bayesian_optimization(X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, n_calls: int = 50) -> Tuple[np.ndarray, List[float], List[float], float]:
    """
    Implémente l'optimisation bayésienne pour la régression logistique.

    Args:
        X (np.ndarray): Matrice des caractéristiques d'entraînement.
        y (np.ndarray): Vecteur des labels d'entraînement.
        X_val (np.ndarray): Matrice des caractéristiques de validation.
        y_val (np.ndarray): Vecteur des labels de validation.
        n_calls (int): Nombre d'appels à la fonction objectif. Par défaut à 50.

    Returns:
        Tuple[np.ndarray, List[float], List[float], float]:
            - weights: Vecteur de poids optimisé.
            - train_losses: Liste des pertes d'entraînement.
            - val_losses: Liste des pertes de validation.
            - convergence_time: Temps de convergence en secondes.
    """
    # Définir l'espace de recherche pour les poids
    space = [Real(-3.0, 3.0, name=f'weight_{i}') for i in range(X.shape[1])]

    # Fonction objectif à minimiser
    @use_named_args(space)
    def objective(**weights_dict):
        weights = np.array([weights_dict[f'weight_{i}'] for i in range(X.shape[1])])
        train_loss = logistic_loss(weights, X, y)
        val_loss = logistic_loss(weights, X_val, y_val)
        return train_loss

    # Exécuter l'optimisation bayésienne
    start_time = time.time()
    result = gp_minimize(objective, space, n_calls=n_calls, random_state=42)
    end_time = time.time()
    convergence_time = end_time - start_time

    # Extraire les poids optimisés
    optimized_weights = np.array(result.x)

    # Calculer les pertes finales
    train_loss = logistic_loss(optimized_weights, X, y)
    val_loss = logistic_loss(optimized_weights, X_val, y_val)

    return optimized_weights, [train_loss], [val_loss], convergence_time