import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Markov_Chain import MarkovChain


def _build_default_case(case_name: str):
    if case_name == "Meteo (Pluie / Nuageux / Soleil)":
        state_names = ["Pluie", "Nuageux", "Soleil"]
        matrix = np.array(
            [
                [0.55, 0.30, 0.15],
                [0.25, 0.50, 0.25],
                [0.10, 0.30, 0.60],
            ],
            dtype=float,
        )
        pi0 = np.array([0.34, 0.33, 0.33], dtype=float)
    else:
        state_names = ["Android", "iOS"]
        matrix = np.array([[0.85, 0.15], [0.10, 0.90]], dtype=float)
        pi0 = np.array([0.60, 0.40], dtype=float)
    return state_names, matrix, pi0


def _row_is_stochastic(row: np.ndarray, tol: float = 1e-6) -> bool:
    return np.all(row >= 0.0) and np.all(row <= 1.0) and np.isclose(np.sum(row), 1.0, atol=tol)


def _distribution_is_valid(dist: np.ndarray, tol: float = 1e-6) -> bool:
    return np.all(dist >= 0.0) and np.all(dist <= 1.0) and np.isclose(np.sum(dist), 1.0, atol=tol)


st.set_page_config(page_title="Markov Dynamic Simulator", layout="wide")

st.title("Simulateur Dynamique de Chaines de Markov")
st.markdown(
    "Ajustez la matrice de transition, la distribution initiale et le nombre de pas. "
    "Les courbes et indicateurs se recalculent automatiquement a chaque changement."
)

# --- Sidebar: Parametres utilisateur ---
st.sidebar.header("Configuration")
example_name = st.sidebar.selectbox(
    "Choisir un scenario",
    ["Meteo (Pluie / Nuageux / Soleil)", "Part de marche (Android / iOS)"],
)
state_names, p_default, pi0_default = _build_default_case(example_name)
n_states = len(state_names)

st.sidebar.subheader("Matrice de transition P")
st.sidebar.caption("Chaque ligne doit sommer a 1.")

P = np.zeros((n_states, n_states), dtype=float)
for i in range(n_states):
    st.sidebar.markdown(f"**Ligne {i + 1} ({state_names[i]})**")
    row_cols = st.sidebar.columns(n_states)
    for j in range(n_states):
        P[i, j] = row_cols[j].number_input(
            f"P({state_names[i]}->{state_names[j]})",
            min_value=0.0,
            max_value=1.0,
            value=float(p_default[i, j]),
            step=0.01,
            key=f"p_{example_name}_{i}_{j}",
        )

st.sidebar.subheader("Distribution initiale pi_0")
pi0 = np.zeros(n_states, dtype=float)
for i in range(n_states):
    pi0[i] = st.sidebar.number_input(
        f"pi_0({state_names[i]})",
        min_value=0.0,
        max_value=1.0,
        value=float(pi0_default[i]),
        step=0.01,
        key=f"pi0_{example_name}_{i}",
    )

n_steps = st.sidebar.number_input("Nombre de pas de temps n", min_value=20, max_value=200000, value=2000, step=10)
target_state_index = st.sidebar.selectbox(
    "Etat cible pour la convergence",
    options=list(range(n_states)),
    format_func=lambda idx: state_names[idx],
)

if not all(_row_is_stochastic(P[i]) for i in range(n_states)):
    st.error("Matrice P invalide: chaque ligne doit contenir des probabilites entre 0 et 1 et sommer a 1.")
    st.stop()

if not _distribution_is_valid(pi0):
    st.error("Distribution initiale invalide: les composantes de pi_0 doivent etre entre 0 et 1 et sommer a 1.")
    st.stop()

# --- Calculs reactifs ---
mc = MarkovChain(state_names, P, pi0)
pi_theorique = mc.get_dist_stationnaire()
trajectory = np.array(mc.simulate(int(n_steps)))

# Frequences empiriques globales
state_counts = np.bincount(trajectory, minlength=len(state_names))
time_share = state_counts / len(trajectory)
abs_error = np.abs(time_share - pi_theorique)

# Courbe de convergence pour un etat cible
cumulative_counts = np.cumsum(trajectory == target_state_index)
frequencies = cumulative_counts / (np.arange(len(trajectory)) + 1)

# --- Layout principal ---
left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("Analyse theorique")
    st.write("Matrice de transition P")
    st.dataframe(P, use_container_width=True)

    st.write("Distribution stationnaire theorique pi")
    st.dataframe(
        {
            "Etat": state_names,
            "pi theorique": np.round(pi_theorique, 6),
        },
        use_container_width=True,
    )

    st.subheader("Heatmap de validation")
    fig_hm, ax_hm = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        P,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=False,
        xticklabels=state_names,
        yticklabels=state_names,
        ax=ax_hm,
    )
    ax_hm.set_xlabel("Etat suivant")
    ax_hm.set_ylabel("Etat actuel")
    ax_hm.set_title("Matrice P")
    st.pyplot(fig_hm)

with right_col:
    st.subheader("Simulation et convergence")

    kpi_1, kpi_2, kpi_3 = st.columns(3)
    kpi_1.metric(
        "Ecart max |empirique - theorie|",
        f"{np.max(abs_error):.4f}",
    )
    kpi_2.metric(
        "Pas simules",
        f"{len(trajectory)}",
    )
    kpi_3.metric(
        "Etat cible convergence",
        state_names[target_state_index],
    )

    st.write("Comparaison KPI: theorie vs pratique")
    st.dataframe(
        {
            "Etat": state_names,
            "pi theorique": np.round(pi_theorique, 6),
            "Temps passe (empirique)": np.round(time_share, 6),
            "Erreur absolue": np.round(abs_error, 6),
        },
        use_container_width=True,
    )

    fig_conv, ax_conv = plt.subplots(figsize=(9, 4))
    ax_conv.plot(
        frequencies,
        label=f"Frequence observee ({state_names[target_state_index]})",
        color="#1f77b4",
    )
    ax_conv.axhline(
        y=pi_theorique[target_state_index],
        color="#d62728",
        linestyle="--",
        label=f"Limite theorique ({state_names[target_state_index]})",
    )
    ax_conv.set_xlabel("Temps")
    ax_conv.set_ylabel("Probabilite")
    ax_conv.set_title("Convergence empirique vers la distribution stationnaire")
    ax_conv.legend()
    st.pyplot(fig_conv)

    zoom_window = min(100, len(trajectory))
    st.subheader(f"Trajectoire (zoom sur les {zoom_window} premiers pas)")
    fig_traj, ax_traj = plt.subplots(figsize=(9, 2.5))
    ax_traj.step(range(zoom_window), trajectory[:zoom_window], where="post", color="#2ca02c")
    ax_traj.set_yticks(range(len(state_names)))
    ax_traj.set_yticklabels(state_names)
    ax_traj.set_xlabel("Temps")
    ax_traj.set_ylabel("Etat")
    ax_traj.set_title("Trajectoire simulee")
    st.pyplot(fig_traj)

st.success("Application reactive active: toute modification dans la barre laterale recalcule les graphiques et KPIs.")