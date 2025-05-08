import mesa
import numpy as np
import random
import math

from crimes import CRIMES
from enums import CriminalStatus
from math import exp, sqrt

from victim_selection import choose_victim
from racketeering import apply_racketeering

from economics import (
    generate_initial_wealth,
    generate_wages_option_a,
    generate_wages_option_b,
    biased_wealth_transfer,
    crime_propensity
)

# Cultural module import
from Culture_language_mechanism import (
    pick_nationality,
    get_languages,
    cultural_favorability,
)

from law_enforcement import probability_of_being_caught, incarcerate, step_incarceration

from crime_decision import (
    wealth_class_bias,
    fraud_access_score,
    filter_eligible_crimes,
    normalize_and_sample,
)

def setup_economy(num_agents, mode="A", r_w=0.05):
    """
    Initializes wealth and wages for all agents.

    Parameters:
        num_agents (int): number of agents
        mode (str): "A" for truncated Pareto wages, "B" for wealth-proportional wages
        r_w (float): wage-to-wealth ratio for Option B

    Returns:
        Tuple of (initial_wealths, wages)
    """
    initial_wealth = generate_initial_wealth(num_agents)

    if mode == "A":
        wages = generate_wages_option_a(num_agents)
    elif mode == "B":
        wages = generate_wages_option_b(initial_wealth, r_w)
    else:
        raise ValueError("Invalid wage mode: choose 'A' or 'B'")

    return initial_wealth, wages


class PersonAgent(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        # Demographics
        self.gender = self.random.choices(["F", "M"], weights=[0.54, 0.46])[0]
        self.age = self.sample_age()
        self.height = self.random.gauss(mu=170, sigma=10)
        self.weight = self.sample_weight_log_normal(self.gender)

        # Muscle mass
        self.muscle_mass_initial = self.sample_muscle_mass_log_normal(self.gender)
        self.muscle_mass = self.compute_muscle_mass(self.age, self.muscle_mass_initial)

        # Social traits
        self.charisma = self.random.uniform(0, 1)

        # Location
        self.location = self.assign_location()

        # Nationality & Language
        self.nationality = pick_nationality()
        self.language_primary, self.language_secondary = get_languages(self.nationality)

        # Criminal status update
        self.criminal_status = CriminalStatus.NON_CRIMINAL
        self.associates = set()

    # Sampling methods

    def sample_age(self):
        age_bins = [
            ((16, 20), 0.10),
            ((21, 25), 0.12),
            ((26, 30), 0.13),
            ((31, 35), 0.14),
            ((36, 40), 0.13),
            ((41, 45), 0.11),
            ((46, 50), 0.10),
            ((51, 55), 0.07),
            ((56, 60), 0.06),
            ((61, 65), 0.04),
        ]
        bins, weights = zip(*age_bins)
        selected_bin = random.choices(bins, weights=weights)[0]
        return random.randint(*selected_bin)

    def sample_weight_log_normal(self, gender):
        mu, sigma = (4.25, 0.17) if gender == "M" else (4.04, 0.20)
        return np.random.lognormal(mean=mu, sigma=sigma)

    def sample_muscle_mass_log_normal(self, gender):
        mu, sigma = (3.5, 0.12) if gender == "M" else (3.09, 0.15)
        return np.random.lognormal(mean=mu, sigma=sigma)

    def compute_muscle_mass(self, age, initial_mass):
        if age < 30:
            return initial_mass
        elif age < 50:
            return initial_mass * math.exp(-0.01 * (age - 30))
        else:
            intermediate_mass = initial_mass * math.exp(-0.01 * 20)
            return intermediate_mass * math.exp(-0.015 * (age - 50))

    def assign_location(self):
        angle = self.random.uniform(0, 2 * np.pi)
        radius = np.sqrt(self.random.uniform(0, 1)) * 3
        return (radius * np.cos(angle), radius * np.sin(angle))

    # Language and Cultural Interactions

    def speaks(self, lang):
        return lang == self.language_primary or lang == self.language_secondary

    def interaction_favorability(self, other_agent):
        return cultural_favorability(self.nationality, other_agent.nationality)

    def status_numeric(self):
        return {
            CriminalStatus.NON_CRIMINAL: 0,
            CriminalStatus.PETTY_CRIMINAL: 1,
            CriminalStatus.ORGANIZED_CRIMINAL: 2,
            CriminalStatus.VORY: 3,
        }.get(self.criminal_status, 0)

    def trait_vector(self):
        return np.array([self.charisma, self.muscle_mass, self.height, self.weight])

    def trait_distance(self, other):
        return np.linalg.norm(self.trait_vector() - other.trait_vector())

    def physical_distance(self, other):
        dx = self.location[0] - other.location[0]
        dy = self.location[1] - other.location[1]
        return sqrt(dx * dx + dy * dy)

    def association_probability(self, other, weights=None, threshold=0.5):
        if weights is None:
            weights = {
                "w0": -2.0,
                "w1": -1.0,
                "w2": -0.5,
                "w3": -1.0,
                "w4": -0.1,
                "w5": -1.0,
            }

        z = (
            weights["w0"]
            + weights["w1"] * (1 - self.interaction_favorability(other))
            + weights["w2"] * self.physical_distance(other)
            + weights["w3"] * abs(self.status_numeric() - other.status_numeric())
            + weights["w4"] * abs(self.age - other.age)
            + weights["w5"] * self.trait_distance(other)
        )
        p = 1 / (1 + exp(-z))
        return p >= threshold

    def add_associate(self, other):
        self.associates.add(other.unique_id)

    def is_associated_with(self, other):
        return other.unique_id in self.associates

    def decide_which_crime(self):
        """
        Decide which crime to commit based on wealth, affiliation, traits, and eligibility filters.
        Returns a string crime name or None.
        """
        candidate_weights = {}

        # --- Step 1: Wealth bias toward petty crimes ---
        petty_bias = wealth_class_bias(self.wealth)
        candidate_weights["theft"] = petty_bias
        candidate_weights["assault"] = petty_bias * 0.5  # less likely

        # --- Step 2: Affiliation and access to organized crime ---
        if self.criminal_status in {CriminalStatus.ORGANIZED_CRIMINAL, CriminalStatus.VORY}:
            fraud_score = fraud_access_score(
                self.wealth, w_f=500, lambda_=0.01, centrality=len(self.associates) + 1
            )
            candidate_weights["racketeering"] = fraud_score

        # --- Step 3: Trait filters ---
        eligible_crimes = filter_eligible_crimes(self, candidate_weights)

        if not eligible_crimes:
            return None

        # --- Step 4: Normalize & sample ---
        return normalize_and_sample(eligible_crimes)

    def attempt_crime(self, s_k, r_k, crime_name="unknown"):
        """
        Attempt a crime and possibly get incarcerated.

        Parameters:
            s_k: minimum sentence (in timesteps)
            r_k: reporting rate (visibility)
            crime_name: optional label
        """
        if hasattr(self, "wealth"):
            p_caught = probability_of_being_caught(self.wealth, s_k, r_k)
            if np.random.random() < p_caught:
                incarcerate(self, s_k)
                print(
                    f"Agent {self.unique_id} caught committing {crime_name} (p={p_caught:.2f})"
                )
            else:
                print(
                    f"Agent {self.unique_id} got away with {crime_name} (p={p_caught:.2f})"
                )
