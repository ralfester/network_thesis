import numpy as np

# PARAMETERS
PARETO_ALPHA_WEALTH = 2.5
PARETO_ALPHA_WAGE = 4.0
WAGE_SCALING_FACTOR = 40
WAGE_CAP = 2000
WEALTH_MIN = 400
WEALTH_MAX = 15000
MIN_WAGE = 20


#  PARETO SAMPLING
def truncated_pareto_sample(alpha, x_min, x_max, size=1):
    u = np.random.uniform(0, 1, size)
    return ((1 - u) * x_min ** (-alpha) + u * x_max ** (-alpha)) ** (-1 / alpha)


# Wealth Initialization
def generate_initial_wealth(num_agents):
    return truncated_pareto_sample(
        PARETO_ALPHA_WEALTH, WEALTH_MIN, WEALTH_MAX, size=num_agents
    )


# Wage Option A
def generate_wages_option_a(num_agents):
    wages = truncated_pareto_sample(
        PARETO_ALPHA_WAGE, WEALTH_MIN, WAGE_CAP, size=num_agents
    )
    return np.minimum(wages, WAGE_CAP)


# Wage Option B
def generate_wages_option_b(initial_wealth, r_w=0.05):
    wages = initial_wealth * r_w
    return np.maximum(wages, MIN_WAGE)  # apply minimum wage floor


def biased_wealth_transfer(agent_i, agent_j, beta=0.05, p_base=0.1, gamma=0.3):
    if agent_i.wealth == 0 and agent_j.wealth == 0:
        return "No transfer: agents have zero wealth"

    w_i, w_j = agent_i.wealth, agent_j.wealth
    richer, poorer = (agent_i, agent_j) if w_i > w_j else (agent_j, agent_i)

    p = p_base * (1 + gamma * (poorer.wealth / richer.wealth))
    p = min(p, 1.0)

    if np.random.random() < p:
        donor, recipient = poorer, richer
    else:
        donor, recipient = richer, poorer

    delta_w = min(beta * donor.wealth, recipient.wealth)
    donor.wealth -= delta_w
    recipient.wealth += delta_w

    return f"{donor.unique_id} to {recipient.unique_id}: transferred {delta_w:.2f}"


def crime_propensity(wealth, w_c=2000, delta=0.01, zeta=0.15, epsilon=1e-2):
    raw = delta * (w_c / (wealth + epsilon)) ** zeta
    return max(0.005, raw)
