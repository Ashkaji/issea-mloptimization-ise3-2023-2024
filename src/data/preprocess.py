# data/preprocess.py

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from typing import Tuple

def load_and_preprocess_data(test_size: float = 0.3, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Charge et prétraite les données du dataset `load_breast_cancer`.

    Args:
        test_size (float): Proportion de l'ensemble de test. Par défaut à 0.3.
        random_state (int): Graine aléatoire pour la reproductibilité. Par défaut à 42.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - X_train: Données d'entraînement.
            - X_val: Données de validation.
            - X_test: Données de test.
            - y_train: Labels d'entraînement.
            - y_val: Labels de validation.
            - y_test: Labels de test.
    """
    # Charger les données
    breast_cancer = load_breast_cancer()
    X, y = breast_cancer.data, breast_cancer.target

    # Diviser les données en ensembles d'entraînement, de validation et de test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)

    # Standardiser les données
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Ajouter un biais pour le terme d'interception
    X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]
    X_val_bias = np.c_[np.ones(X_val.shape[0]), X_val]
    X_test_bias = np.c_[np.ones(X_test.shape[0]), X_test]

    return X_train_bias, X_val_bias, X_test_bias, y_train, y_val, y_test