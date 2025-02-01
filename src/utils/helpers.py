# utils/helpers.py

import numpy as np
from typing import Tuple

def logistic_loss(weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """
    Calcule la perte logistique (entropie croisée binaire).

    Args:
        weights (np.ndarray): Vecteur de poids du modèle.
        X (np.ndarray): Matrice des caractéristiques.
        y (np.ndarray): Vecteur des labels.

    Returns:
        float: La valeur de la perte logistique.
    """
    predictions = 1 / (1 + np.exp(-X @ weights))
    predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
    return -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))

def soft_thresholding(weight: float, lambda_: float) -> float:
    """
    Applique le soft thresholding pour la régularisation L1.

    Args:
        weight (float): Poids à régulariser.
        lambda_ (float): Paramètre de régularisation.

    Returns:
        float: Poids après soft thresholding.
    """
    return np.sign(weight) * max(abs(weight) - lambda_, 0)