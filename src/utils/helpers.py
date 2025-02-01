# utils/helpers.py

import numpy as np
import matplotlib.pyplot as plt

RANDOM_STATE = 42


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


def plot_losses(train_losses, val_losses, title):
    """
    Trace les courbes de perte pour l'entraînement et la validation.

    Args:
        train_losses (list): Liste des pertes d'entraînement.
        val_losses (list): Liste des pertes de validation.
        title (str): Titre du graphique.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Perte Entraînement', color='blue')
    plt.plot(val_losses, label='Perte Validation', color='orange')
    plt.title(f'Courbes de Perte : {title}')
    plt.xlabel('Itérations')
    plt.ylabel('Perte')
    plt.legend()
    plt.grid()
    plt.show()


def calculate_generalization_error(trained_weights, X_test_bias, y_test):
    """Calcule l'erreur de généralisation du modèle sur l'ensemble de test"""
    y_pred_test = np.round(1 / (1 + np.exp(-X_test_bias @ trained_weights)))
    return np.mean(y_pred_test != y_test)