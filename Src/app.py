import streamlit as st
import numpy as np
import pandas as pd
from Markov_Chain import MarkovChain

# Section 1: Streamlit page setup.
st.set_page_config(page_title="Markov Simulator", layout="centered")

st.title("Simulateur de Chaine de Markov")
st.caption("Version simplifiee: choisissez un scenario, lancez la simulation, et voyez la convergence.")

# Section 2: Scenario data (no helper function; direct configuration in app).
scenarios = {
    "Meteo (Pluie / Nuageux / Soleil)": {
        "state_names": ["Pluie", "Nuageux", "Soleil"],
        "P": np.array(
            [
                [0.55, 0.30, 0.15],
                [0.25, 0.50, 0.25],
                [0.10, 0.30, 0.60],
            ],
            dtype=float,
        ),
        "pi0": np.array([0.34, 0.33, 0.33], dtype=float),
    },
    "Part de marche (Android / iOS)": {
        "state_names": ["Android", "iOS"],
        "P": np.array([[0.85, 0.15], [0.10, 0.90]], dtype=float),
        "pi0": np.array([0.60, 0.40], dtype=float),
    },
}

# Section 3: User controls.
scenario_name = st.selectbox(
    "Scenario",
    list(scenarios.keys()),
)

state_names = scenarios[scenario_name]["state_names"]
P = scenarios[scenario_name]["P"]
pi0 = scenarios[scenario_name]["pi0"]
n_steps = st.slider("Nombre de pas", min_value=100, max_value=20000, value=5000, step=100)
target_state_index = st.selectbox(
    "Etat a suivre",
    options=list(range(len(state_names))),
    format_func=lambda idx: state_names[idx],
)

# Section 4: Editable transition matrix.
st.subheader("Modifier la matrice de transition")
st.caption("Chaque valeur doit etre entre 0 et 1, et chaque ligne doit sommer a 1.")
matrix_df = pd.DataFrame(P, index=state_names, columns=state_names)
edited_matrix_df = st.data_editor(
    matrix_df,
    use_container_width=True,
    num_rows="fixed",
    key=f"matrix_editor_{scenario_name}",
)

try:
    P = edited_matrix_df.to_numpy(dtype=float)
except ValueError:
    st.error("La matrice contient des valeurs non numeriques.")
    st.stop()

if np.any(P < 0) or np.any(P > 1):
    st.error("Toutes les probabilites de transition doivent etre entre 0 et 1.")
    st.stop()

row_sums = P.sum(axis=1)
if not np.allclose(row_sums, 1.0, atol=1e-6):
    st.error(
        "Chaque ligne de la matrice doit sommer a 1. "
        f"Sommes actuelles: {np.round(row_sums, 4).tolist()}"
    )
    st.stop()

# Section 5: Core Markov computations using your class methods.
mc = MarkovChain(state_names, P, pi0)
pi_theorique = mc.get_dist_stationnaire()
trajectory = np.array(mc.simulate(int(n_steps)))

# Section 6: Derived metrics displayed in the table.
state_counts = np.bincount(trajectory, minlength=len(state_names))
time_share = state_counts / len(trajectory)
abs_error = np.abs(time_share - pi_theorique)

# Section 7: Summary table.
st.subheader("Resultats")
st.dataframe(
    {
        "Etat": state_names,
        "Pi theorique": np.round(pi_theorique, 4),
        "Frequence simulee": np.round(time_share, 4),
        "Erreur": np.round(abs_error, 4),
    },
    use_container_width=True,
)

# Section 8: Convergence visualization from class function.
fig_conv = mc.plot_convergence(trajectory, target_state_index, pi_theorique[target_state_index])
st.pyplot(fig_conv)

st.subheader("Matrice de transition")
# Section 9: Transition matrix heatmap from class function.
fig_heatmap = mc.plot_heatmap()
st.pyplot(fig_heatmap)

zoom_window = min(120, len(trajectory))
st.subheader(f"Trajectoire (premiers {zoom_window} pas)")
# Section 10: Trajectory plot from class function.
fig_traj = mc.plot_trajectoire(trajectory[:zoom_window])
st.pyplot(fig_traj)