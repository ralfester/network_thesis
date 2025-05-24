import numpy as np
from enums import CriminalStatus


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


import numpy as np


import numpy as np

def probability_of_being_caught(wealth, report_rate):
    """
    Bribe-driven capture model:
    - 30% of agent's wealth is offered as a bribe.
    - Bribes < 200 rubles do nothing.
    - Bribes ≥ 200 gain effectiveness from 0.25 to 0.95, saturating at 2000.
    - Final caught probability = report_rate - bribe_effectiveness
    """

    bribe = 0.2 * wealth
    threshold = 500
    saturation = 4000

    base_success = 0.25
    max_success = 0.95

    if bribe < threshold:
        bribe_effectiveness = 0.0
    else:
        k = -np.log(1 - ((max_success - base_success) / (max_success))) / (saturation - threshold)
        bribe_effectiveness = base_success + (max_success - base_success) * (1 - np.exp(-k * (bribe - threshold)))

    p_caught = report_rate - bribe_effectiveness
    return max(0.05, p_caught)


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
