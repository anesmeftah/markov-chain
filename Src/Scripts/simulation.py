from pathlib import Path
import numpy as np
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Src.Markov_Chain import MarkovChain

P_meteo = np.array([[0.7, 0.3], 
                    [0.2, 0.8]])
# Distribution initiale : 30% Rain, 70% No_Rain 
pi0 = [0.3, 0.7]
mc_meteo = MarkovChain(["No_Rain", "Rain"], P_meteo, pi0)




# Simulation
trajectoire = mc_meteo.simulate(n_steps=200)
print(f"Trajectoire simulée : {trajectoire}")

mc_meteo.plot_trajectoire(trajectoire)

mc_meteo.validate_simulation(n_steps=10000)


mc_meteo.plot_convergence(trajectoire , 1, 0.4)

# 3. Tracer la Heatmap
mc_meteo.plot_heatmap()