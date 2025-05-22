import numpy as np
from enums import CriminalStatus


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def probability_of_being_caught(
    wealth, severity, report_rate, s_max=6, alpha=0.5, lambda_=1.5, mu=0.005, theta=4.0
):
    bribe_component = 1 - sigmoid(mu * np.log1p(wealth) - theta)
    severity_component = (np.exp(lambda_ * severity) - 1) / (
        np.exp(lambda_ * s_max) - 1
    )
    weighted_visibility = alpha * severity_component + (1 - alpha) * report_rate
    p = bribe_component * weighted_visibility
    return max(p, 0.01)  # ensure floor chance of 1%


def incarcerate(agent, sentence_length):
    """Sets incarceration state for a caught agent"""
    agent.incarcerated = True
    agent.sentence_timer = sentence_length


def step_incarceration(agent):
    """Decrement sentence and release agent if time is fully served"""
    if agent.incarcerated:
        agent.sentence_timer = max(0, agent.sentence_timer - 1)
        if agent.sentence_timer == 0:
            agent.incarcerated = False
            if np.random.random() < 0.1:  # desistance
                agent.desisted = True
                if agent.criminal_status in {
                    CriminalStatus.ORGANIZED_CRIMINAL,
                    CriminalStatus.VORY,
                }:
                    agent.immune = True
