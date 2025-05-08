import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def wealth_class_bias(w_i, w_s=100, epsilon=1e-2):
    return w_s / (w_i + epsilon)


def fraud_access_score(w_i, w_f=500, lambda_=0.01, centrality=1.0):
    return sigmoid(lambda_ * (w_i - w_f)) * centrality


def filter_eligible_crimes(agent, candidate_crimes, max_distance=3.0):
    eligible = {}

    for crime in candidate_crimes:
        if crime == "robbery" and agent.muscle_mass < 30:  # τ_force
            continue
        if crime == "bribery" and agent.charisma < 0.7:  # τ_charisma
            continue
        # Additional gates could be added here (e.g., distance, affiliation)
        eligible[crime] = candidate_crimes[crime]

    return eligible


def normalize_and_sample(weights_dict):
    keys = list(weights_dict.keys())
    values = np.array(list(weights_dict.values()), dtype=np.float64)
    probs = values / values.sum()
    return np.random.choice(keys, p=probs)
