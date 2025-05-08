import numpy as np


def intimidation_factor(muscle_i, muscle_j, epsilon=1e-2):
    return max(0.1, muscle_i / (muscle_j + epsilon))


def distance_decay(distance, tau=30):
    k = np.log(2) / tau
    return np.exp(-k * distance)


def victim_selection_prob(agent_i, agent_j, tau=30):
    if agent_j.incarcerated or agent_j.desisted or getattr(agent_j, "immune", False):
        return 0  # not eligible

    d = agent_i.physical_distance(agent_j)
    w = agent_j.wealth
    intimidation = intimidation_factor(agent_i.muscle_mass, agent_j.muscle_mass)

    return distance_decay(d, tau) * np.log1p(w) * intimidation


def choose_victim(agent_i, population, tau=30):
    scores = []
    candidates = []

    for agent_j in population:
        if agent_j.unique_id == agent_i.unique_id:
            continue
        p = victim_selection_prob(agent_i, agent_j, tau)
        if p > 0:
            scores.append(p)
            candidates.append(agent_j)

    if not candidates:
        return None

    probs = np.array(scores) / np.sum(scores)
    return np.random.choice(candidates, p=probs)
