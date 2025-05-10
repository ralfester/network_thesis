import numpy as np


def robbery_gain(victim_wealth, alpha=0.1, murder_chance=0.00022):
    """
    Returns monetary gain and a note indicating if murder occurred.
    """
    if np.random.random() < murder_chance:
        return 0, True
    return alpha * victim_wealth, False


def fraud_gain(victim_wealth, beta=0.2):
    return beta * victim_wealth


def racketeering_gain(num_victims, R=50):
    return R * num_victims
