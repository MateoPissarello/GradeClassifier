"""Q-Learning simulation for the GradeClassifier project.

The script builds a small tabular reinforcement-learning environment where the
states correspond to the four K-Means clusters that were interpreted in the
original exploratory notebook.  Each action represents an academic
intervention.  Transition dynamics and rewards are hand-crafted based on the
cluster narratives so that we can reason about a plausible optimal policy even
without access to the remote Kaggle dataset.

Running the module will train a Q-Learning agent with a decaying epsilon-greedy
strategy and will print the learnt Q-table, the greedy policy, and a short
analysis of the policy.
"""
from __future__ import annotations

import dataclasses
import random
from typing import Dict, List, Tuple


StateName = str
ActionName = str


@dataclasses.dataclass(frozen=True)
class StateProfile:
    """Summary statistics describing each cluster/state."""

    label: str
    avg_gpa: float
    avg_absences: float
    description: str


class AcademicInterventionEnv:
    """Small tabular environment that mimics transitions between clusters.

    The environment is intentionally light-weight: transitions were designed
    with simple heuristics extracted from the qualitative cluster analysis.  A
    transition brings the student to another cluster with a probability that
    depends on the action performed.  Rewards encourage movements towards
    higher GPA and fewer absences.
    """

    def __init__(self) -> None:
        # The four states come directly from the cluster interpretation cells.
        self.state_profiles: Dict[StateName, StateProfile] = {
            "cluster_0": StateProfile(
                label="Cluster 0 – Bajo estudio + más ausencias (deportivos)",
                avg_gpa=2.1,
                avg_absences=20,
                description=(
                    "Estudiantes con poco tiempo de estudio y muchas ausencias; "
                    "actividades deportivas altas y riesgo académico."),
            ),
            "cluster_1": StateProfile(
                label="Cluster 1 – Muy estudioso + menos ausencias (musicales)",
                avg_gpa=3.8,
                avg_absences=5,
                description=(
                    "Perfil académico sobresaliente con apoyo parental alto y "
                    "participación musical."),
            ),
            "cluster_2": StateProfile(
                label="Cluster 2 – Mucho estudio + apoyo medio",
                avg_gpa=3.5,
                avg_absences=8,
                description=(
                    "Estudiantes con gran dedicación al estudio y buen apoyo "
                    "familiar, pero menor participación extracurricular."),
            ),
            "cluster_3": StateProfile(
                label="Cluster 3 – Poco estudio + más ausencias (voluntariado alto)",
                avg_gpa=2.4,
                avg_absences=18,
                description=(
                    "Mayor involucramiento en voluntariado, pero bajo desempeño "
                    "académico y ausencias elevadas."),
            ),
        }

        # Actions correspond to the intervention rules described in the
        # recommendation section of the notebook.
        self.actions: List[ActionName] = [
            "intensive_tutoring",  # Refuerza estudio y controla ausencias.
            "attendance_monitoring",  # Foco en asistencia y reportes a padres.
            "balanced_support",  # Combinación de mentoría y seguimiento.
            "wellbeing_program",  # Reduce burnout, ajusta carga extracurricular.
        ]

        # Heuristic transition model: {state: {action: [(prob, next_state), ...]}}
        # Probabilities sum to 1 for each (state, action).
        self.transition_model: Dict[
            StateName, Dict[ActionName, List[Tuple[float, StateName]]]
        ] = {
            "cluster_0": {
                "intensive_tutoring": [
                    (0.55, "cluster_1"),
                    (0.25, "cluster_2"),
                    (0.15, "cluster_0"),
                    (0.05, "cluster_3"),
                ],
                "attendance_monitoring": [
                    (0.45, "cluster_2"),
                    (0.3, "cluster_1"),
                    (0.2, "cluster_0"),
                    (0.05, "cluster_3"),
                ],
                "balanced_support": [
                    (0.4, "cluster_1"),
                    (0.3, "cluster_2"),
                    (0.2, "cluster_3"),
                    (0.1, "cluster_0"),
                ],
                "wellbeing_program": [
                    (0.3, "cluster_2"),
                    (0.35, "cluster_3"),
                    (0.2, "cluster_1"),
                    (0.15, "cluster_0"),
                ],
            },
            "cluster_1": {
                "intensive_tutoring": [
                    (0.7, "cluster_1"),
                    (0.15, "cluster_2"),
                    (0.1, "cluster_0"),
                    (0.05, "cluster_3"),
                ],
                "attendance_monitoring": [
                    (0.6, "cluster_1"),
                    (0.2, "cluster_2"),
                    (0.1, "cluster_3"),
                    (0.1, "cluster_0"),
                ],
                "balanced_support": [
                    (0.65, "cluster_1"),
                    (0.2, "cluster_2"),
                    (0.1, "cluster_3"),
                    (0.05, "cluster_0"),
                ],
                "wellbeing_program": [
                    (0.5, "cluster_1"),
                    (0.3, "cluster_2"),
                    (0.15, "cluster_3"),
                    (0.05, "cluster_0"),
                ],
            },
            "cluster_2": {
                "intensive_tutoring": [
                    (0.5, "cluster_2"),
                    (0.25, "cluster_1"),
                    (0.15, "cluster_0"),
                    (0.1, "cluster_3"),
                ],
                "attendance_monitoring": [
                    (0.45, "cluster_2"),
                    (0.3, "cluster_1"),
                    (0.15, "cluster_3"),
                    (0.1, "cluster_0"),
                ],
                "balanced_support": [
                    (0.4, "cluster_1"),
                    (0.35, "cluster_2"),
                    (0.15, "cluster_3"),
                    (0.1, "cluster_0"),
                ],
                "wellbeing_program": [
                    (0.35, "cluster_2"),
                    (0.3, "cluster_1"),
                    (0.2, "cluster_3"),
                    (0.15, "cluster_0"),
                ],
            },
            "cluster_3": {
                "intensive_tutoring": [
                    (0.5, "cluster_2"),
                    (0.3, "cluster_1"),
                    (0.15, "cluster_3"),
                    (0.05, "cluster_0"),
                ],
                "attendance_monitoring": [
                    (0.45, "cluster_2"),
                    (0.25, "cluster_1"),
                    (0.2, "cluster_3"),
                    (0.1, "cluster_0"),
                ],
                "balanced_support": [
                    (0.35, "cluster_2"),
                    (0.3, "cluster_1"),
                    (0.25, "cluster_3"),
                    (0.1, "cluster_0"),
                ],
                "wellbeing_program": [
                    (0.3, "cluster_2"),
                    (0.25, "cluster_1"),
                    (0.35, "cluster_3"),
                    (0.1, "cluster_0"),
                ],
            },
        }

        # Prior distribution for starting clusters (approximated from narrative).
        self.initial_distribution: Dict[StateName, float] = {
            "cluster_0": 0.3,
            "cluster_1": 0.25,
            "cluster_2": 0.25,
            "cluster_3": 0.2,
        }

        self.states: List[StateName] = list(self.state_profiles)

    # ------------------------------------------------------------------
    # Environment mechanics
    # ------------------------------------------------------------------
    def reset(self) -> StateName:
        """Sample an initial state following the prior distribution."""

        rnd = random.random()
        cumulative = 0.0
        for state, prob in self.initial_distribution.items():
            cumulative += prob
            if rnd <= cumulative:
                return state
        # Fallback – due to floating point rounding.
        return self.states[-1]

    def sample_transition(self, state: StateName, action: ActionName) -> StateName:
        """Sample next state from the transition probabilities."""

        rnd = random.random()
        cumulative = 0.0
        for prob, next_state in self.transition_model[state][action]:
            cumulative += prob
            if rnd <= cumulative:
                return next_state
        return self.transition_model[state][action][-1][1]

    def reward(self, state: StateName, next_state: StateName) -> float:
        """Reward encourages higher GPA and lower absences."""

        current = self.state_profiles[state]
        nxt = self.state_profiles[next_state]
        gpa_gain = nxt.avg_gpa - current.avg_gpa
        absence_delta = current.avg_absences - nxt.avg_absences
        # Scale rewards to keep them bounded.
        return gpa_gain + 0.05 * absence_delta

    def step(self, state: StateName, action: ActionName) -> Tuple[StateName, float]:
        next_state = self.sample_transition(state, action)
        return next_state, self.reward(state, next_state)


def q_learning(
    env: AcademicInterventionEnv,
    episodes: int = 5000,
    max_steps: int = 12,
    alpha: float = 0.2,
    gamma: float = 0.9,
    epsilon_start: float = 0.3,
    epsilon_end: float = 0.05,
) -> List[List[float]]:
    """Standard tabular Q-learning with linear epsilon decay."""

    num_states = len(env.states)
    num_actions = len(env.actions)
    q_table = [[0.0 for _ in range(num_actions)] for _ in range(num_states)]

    def epsilon_for_episode(ep: int) -> float:
        frac = ep / max(1, episodes - 1)
        return epsilon_start + frac * (epsilon_end - epsilon_start)

    state_to_index = {s: i for i, s in enumerate(env.states)}
    action_to_index = {a: i for i, a in enumerate(env.actions)}

    for episode in range(episodes):
        state = env.reset()
        eps = epsilon_for_episode(episode)

        for _ in range(max_steps):
            s_idx = state_to_index[state]
            if random.random() < eps:
                action = random.choice(env.actions)
            else:
                row = q_table[s_idx]
                best_idx = max(range(num_actions), key=lambda idx: row[idx])
                action = env.actions[best_idx]

            a_idx = action_to_index[action]
            next_state, reward = env.step(state, action)
            ns_idx = state_to_index[next_state]

            best_next = max(q_table[ns_idx])
            td_target = reward + gamma * best_next
            td_error = td_target - q_table[s_idx][a_idx]
            q_table[s_idx][a_idx] += alpha * td_error

            state = next_state

    return q_table


def greedy_policy(q_table: List[List[float]], env: AcademicInterventionEnv) -> Dict[str, str]:
    policy: Dict[str, str] = {}
    for state_idx, state in enumerate(env.states):
        row = q_table[state_idx]
        best_action_idx = max(range(len(row)), key=lambda idx: row[idx])
        policy[state] = env.actions[best_action_idx]
    return policy


def simulate_policy(
    policy: Dict[str, str],
    env: AcademicInterventionEnv,
    episodes: int = 300,
    max_steps: int = 12,
) -> Tuple[float, float]:
    """Evaluate the greedy policy by Monte Carlo simulation."""

    total_reward = 0.0
    successful_transitions = 0

    for _ in range(episodes):
        state = env.reset()
        for _ in range(max_steps):
            action = policy[state]
            next_state, reward = env.step(state, action)
            total_reward += reward
            if env.state_profiles[next_state].avg_gpa > env.state_profiles[state].avg_gpa:
                successful_transitions += 1
            state = next_state

    avg_reward = total_reward / (episodes * max_steps)
    success_rate = successful_transitions / (episodes * max_steps)
    return avg_reward, success_rate


def describe_policy(policy: Dict[str, str], env: AcademicInterventionEnv) -> str:
    lines = ["Política óptima encontrada:"]
    for state_key in env.states:
        profile = env.state_profiles[state_key]
        action = policy[state_key]
        lines.append(
            f"- {profile.label}: aplicar **{action.replace('_', ' ')}**.\n"
            f"  {profile.description}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    random.seed(13)

    environment = AcademicInterventionEnv()
    q = q_learning(environment)
    policy = greedy_policy(q, environment)
    avg_reward, success_rate = simulate_policy(policy, environment)

    print("Q-table (estado x acción):")
    header = "\\t" + "\\t".join(environment.actions)
    print(header)
    for idx, state in enumerate(environment.states):
        row = "\\t".join(f"{value:6.3f}" for value in q[idx])
        print(f"{state}\\t{row}")

    print()
    print(describe_policy(policy, environment))
    print()
    print(
        f"Recompensa promedio por paso: {avg_reward:.3f}. "
        f"Tasa de transiciones con mejora de GPA: {success_rate:.2%}."
    )
