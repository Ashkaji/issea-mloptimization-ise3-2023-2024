# optimization/pso.py

import numpy as np
import time
from typing import Tuple, List
from ..utils.helpers import logistic_loss

def particle_swarm_optimization(X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, n_particles: int = 30, n_iterations: int = 100, w: float = 0.5, c1: float = 1.0, c2: float = 1.0) -> Tuple[np.ndarray, List[float], List[float], float]:
    """
    Implémente l'algorithme PSO (Particle Swarm Optimization).

    Args:
        X (np.ndarray): Matrice des caractéristiques d'entraînement.
        y (np.ndarray): Vecteur des labels d'entraînement.
        X_val (np.ndarray): Matrice des caractéristiques de validation.
        y_val (np.ndarray): Vecteur des labels de validation.
        n_particles (int): Nombre de particules. Par défaut à 30.
        n_iterations (int): Nombre maximal d'itérations. Par défaut à 100.
        w (float): Paramètre d'inertie. Par défaut à 0.5.
        c1 (float): Paramètre cognitif. Par défaut à 1.0.
        c2 (float): Paramètre social. Par défaut à 1.0.

    Returns:
        Tuple[np.ndarray, List[float], List[float], float]:
            - weights: Vecteur de poids optimisé.
            - train_losses: Liste des pertes d'entraînement.
            - val_losses: Liste des pertes de validation.
            - convergence_time: Temps de convergence en secondes.
    """
    n_features = X.shape[1]
    particles = np.random.randn(n_particles, n_features) * 0.01
    velocities = np.random.randn(n_particles, n_features) * 0.01
    personal_best_positions = np.copy(particles)
    personal_best_scores = np.array([logistic_loss(p, X, y) for p in personal_best_positions])

    global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
    global_best_score = min(personal_best_scores)

    train_losses = []
    val_losses = []

    start_time = time.time()

    for _ in range(n_iterations):
        for i in range(n_particles):
            r1, r2 = np.random.rand(2)
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (personal_best_positions[i] - particles[i]) +
                             c2 * r2 * (global_best_position - particles[i]))
            particles[i] += velocities[i]

            current_score = logistic_loss(particles[i], X, y)

            if current_score < personal_best_scores[i]:
                personal_best_scores[i] = current_score
                personal_best_positions[i] = particles[i]

                if current_score < global_best_score:
                    global_best_score = current_score
                    global_best_position = particles[i]

        train_loss = global_best_score
        val_loss = logistic_loss(global_best_position, X_val, y_val)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    end_time = time.time()
    convergence_time = end_time - start_time

    return global_best_position, train_losses, val_losses, convergence_time