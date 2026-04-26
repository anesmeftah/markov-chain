import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class MarkovChain:
    def __init__(self , etats : np.ndarray , matrice_transition : np.ndarray , pi_init = 0.5):
        """Initialise la chaine de Markov avec les états et la matrice de transition."""
        
        self.matrice_transition = matrice_transition
        if not np.allclose(np.sum(self.matrice_transition, axis=1), 1): # verifier si la somme des probas d'une ligne = a 1
            raise ValueError("Chaque ligne de la matrice doit sommer à 1.")
        self.etats = etats
        self.num_etats = len(etats)
        self.pi_0 = pi_init

    def simulate(self , n_steps):
        """
        Simulation de la trajectoire d'une chaine
        Resultat : un vecteur d'etats
        """
        # Choisir l'état de depart.
        current_state = np.random.choice(self.num_etats , p=self.pi_0)
        trajectoire = [current_state]

        for _ in range(n_steps):
            current_state = np.random.choice(self.num_etats , p=self.matrice_transition[current_state])
            trajectoire.append(current_state)

        return trajectoire
    
    def get_dist_stationnaire(self):
        """ Calculer la distribution Stationnaire"""

        evals, evecs = np.linalg.eig(self.matrice_transition.T)
        # On cherche la valeur propre la plus proche de 1
        stationary = np.real(evecs[:, np.argmin(np.abs(evals - 1.0))])
        return stationary / stationary.sum() # Normalisation pour que la somme = 1 

    def get_distribution_at_n(self, initial_dist, n):
        """
        Calcule la distribution exacte après n pas
        """
        Pn = np.linalg.matrix_power(self.matrice_transition, n) # Utilisation de la puissance de matrice [cite: 46]
        return np.dot(initial_dist, Pn)

    def validate_simulation(self, n_steps=10000):
        """
        Compare la fréquence empirique des états avec la théorie.
        """
        # 1. Lancer une longue simulation
        trajectory = self.simulate(n_steps)
        
        # 2. Calculer les fréquences observées
        unique, counts = np.unique(trajectory, return_counts=True)
        frequencies = counts / n_steps
        
        # 3. Récupérer la distribution théorique
        theoretical_pi = self.get_dist_stationnaire()
        
        print(f"--- Validation sur {n_steps} pas ---")
        for i, name in enumerate(self.etats):
            diff = abs(frequencies[i] - theoretical_pi[i])
            print(f"{name}:")
            print(f"  Simulé : {frequencies[i]:.4f}")
            print(f"  Théorie : {theoretical_pi[i]:.4f}")
            print(f"  Écart   : {diff:.4f}")
        
        return frequencies, theoretical_pi
    
    def plot_trajectoire(self , trajectoire):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.step(range(len(trajectoire)), trajectoire, where='post')
        ax.set_yticks(range(len(self.etats)))
        ax.set_yticklabels(self.etats)
        ax.set_xlabel("Temps (Pas)")
        ax.set_ylabel("État")
        ax.set_title("Simulation de la trajectoire de la chaîne de Markov")
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        return fig


    def plot_convergence(self , trajectoire , target_state_index , theoretical_value):

        cumulative_counts = np.cumsum(np.array(trajectoire) == target_state_index)
        frequencies = cumulative_counts / (np.arange(len(trajectoire)) + 1)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(frequencies, label="Fréquence observée (Empirique)")
        ax.axhline(y=theoretical_value, color='r', linestyle='--', label="Valeur théorique (π)")
        ax.set_xlabel("Nombre d'itérations")
        ax.set_ylabel("Probabilité")
        ax.set_title("Convergence vers la distribution stationnaire")
        ax.legend()
        return fig

    def plot_heatmap(self):
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            self.matrice_transition,
            annot=True,
            xticklabels=self.etats,
            yticklabels=self.etats,
            cmap="Blues",
            fmt=".2f",
            ax=ax,
        )
        ax.set_title("Matrice de Transition P")
        ax.set_ylabel("État Actuel")
        ax.set_xlabel("État Suivant")
        return fig



        
    
