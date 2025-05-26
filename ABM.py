import mesa
import numpy as np
import random
import math

from crimes import CRIMES
from enums import CriminalStatus
from math import exp, sqrt

from victim_selection import choose_victim
from racketeering import apply_racketeering
from criminal_progression import update_status

from crime_rewards import robbery_gain, fraud_gain, racketeering_gain

from economics import (
    generate_initial_wealth,
    generate_wages_option_a,
    generate_wages_option_b,
    biased_wealth_transfer,
    crime_propensity,
)

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


# --- Sampling Functions ---


def sample_gender():
    return np.random.choice(["F", "M"], p=[0.54, 0.46])


# --- Economy Setup ---


def setup_economy(num_agents, mode="A", r_w=0.05):
    initial_wealth = generate_initial_wealth(num_agents)
    if mode == "A":
        wages = generate_wages_option_a(num_agents)
    elif mode == "B":
        wages = generate_wages_option_b(initial_wealth, r_w)
    else:
        raise ValueError("Invalid wage mode: choose 'A' or 'B'")
    return initial_wealth, wages


# --- Agent Definition ---


class PersonAgent:
    def __init__(self, unique_id, model):
        self.unique_id = unique_id
        self.model = model

        # Demographics
        self.gender = sample_gender()
        self.age = self.sample_age()
        self.height = random.gauss(mu=170, sigma=10)
        self.weight = self.sample_weight_log_normal(self.gender)

        # Muscle mass
        self.muscle_mass_initial = self.sample_muscle_mass_log_normal(self.gender)
        self.muscle_mass = self.compute_muscle_mass(self.age, self.muscle_mass_initial)

        # Social traits
        self.charisma = random.uniform(0, 1)

        # Location
        self.location = self.assign_location()

        # Cultural identity
        self.nationality = pick_nationality()
        self.language_primary, self.language_secondary = get_languages(self.nationality)

        # Criminal network
        self.criminal_status = CriminalStatus.NON_CRIMINAL
        self.associates = set()

    # --- Sampling Methods ---

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
            intermediate = initial_mass * math.exp(-0.01 * 20)
            return intermediate * math.exp(-0.015 * (age - 50))

    def assign_location(self):
        angle = random.uniform(0, 2 * np.pi)
        radius = np.sqrt(random.uniform(0, 1)) * 3
        return (radius * np.cos(angle), radius * np.sin(angle))

    # --- Cultural + Social ---

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

    # --- Distance + Association ---

    def trait_vector(self):
        return np.array([self.charisma, self.muscle_mass, self.height, self.weight])

    def trait_distance(self, other):
        return np.linalg.norm(self.trait_vector() - other.trait_vector())

    def physical_distance(self, other):
        dx = self.location[0] - other.location[0]
        dy = self.location[1] - other.location[1]
        return sqrt(dx * dx + dy * dy)

    def association_probability(self, other, weights=None, threshold=0.15):
        if weights is None:
            weights = {
                "w0": 1.0,  # base bias → lifted from negative
                "w1": -0.3,  # cultural difference (softer)
                "w2": -0.2,  # physical distance (softened)
                "w3": -0.3,  # status difference
                "w4": -0.02,  # age difference
                "w5": -0.3,  # trait distance
            }

        # Strong bonus for criminal-to-criminal bonding
        if (
            self.criminal_status != CriminalStatus.NON_CRIMINAL
            and other.criminal_status != CriminalStatus.NON_CRIMINAL
        ):
            weights["w0"] += 2.0  # major boost to base chance

        z = (
            weights["w0"]
            + weights["w1"] * (1 - self.interaction_favorability(other))
            + weights["w2"] * self.physical_distance(other)
            + weights["w3"] * abs(self.status_numeric() - other.status_numeric())
            + weights["w4"] * abs(self.age - other.age)
            + weights["w5"] * self.trait_distance(other)
        )

        return 1 / (1 + exp(-z)) >= threshold

    def add_associate(self, other):
        self.associates.add(other.unique_id)

    def is_associated_with(self, other):
        return other.unique_id in self.associates

    # --- Crime Logic ---

    def step_form_associations(self, all_agents):
    # Only criminals form associations
        if self.criminal_status == CriminalStatus.NON_CRIMINAL:
            return

        for other in all_agents:
            if other.unique_id == self.unique_id:
                continue

            # Also skip if other is not a criminal
            if other.criminal_status == CriminalStatus.NON_CRIMINAL:
                continue

            if not self.is_associated_with(other) and self.association_probability(other):
                self.add_associate(other)
                other.add_associate(self)

    def decide_which_crime(self):
        candidate_weights = {}

        petty_bias = wealth_class_bias(self.wealth)
        candidate_weights["theft"] = petty_bias
        candidate_weights["assault"] = petty_bias * 0.5

        if self.criminal_status in {
            CriminalStatus.ORGANIZED_CRIMINAL,
            CriminalStatus.VORY,
        }:
            fraud_score = fraud_access_score(
                self.wealth, w_f=500, lambda_=0.01, centrality=len(self.associates) + 1
            )
            candidate_weights["racketeering"] = fraud_score

        eligible_crimes = filter_eligible_crimes(self, candidate_weights)
        return normalize_and_sample(eligible_crimes) if eligible_crimes else None

    def attempt_crime(self, s_k, r_k, crime_name="unknown", victim=None):
        if not hasattr(self, "wealth"):
            return

        p_caught = probability_of_being_caught(self.wealth, r_k)
        caught = np.random.random() < p_caught

        if caught:
            sentence_years = s_k + np.random.randint(0, 3)
            sentence_months = sentence_years * 12
            incarcerate(self, sentence_months)
            print(
                f"Agent {self.unique_id} caught committing {crime_name} (p={p_caught:.2f})"
            )
        else:
            print(
                f"Agent {self.unique_id} got away with {crime_name} (p={p_caught:.2f})"
            )

            gain = 0
            if crime_name in {"theft", "assault", "robbery"} and victim:
                gain, murder_flag = robbery_gain(victim.wealth)
                if murder_flag:
                    print(f"Agent {self.unique_id} escalated to murder!")
                    gain = 0
            elif crime_name == "fraud" and victim:
                gain = fraud_gain(victim.wealth)
            elif crime_name == "racketeering":
                num_victims = getattr(self, "num_racket_victims", 0)
                gain = racketeering_gain(num_victims)

            self.wealth += gain

        update_status(self, s_k, w_k=1.3 if crime_name == "assault" else 1.0)

    def step_criminal_activity(self, all_agents):
        if np.random.random() > crime_propensity(self.wealth):
            return

        crime = self.decide_which_crime()
        if not crime:
            return

        victim = choose_victim(self, all_agents, crime_type=crime)
        if not victim:
            return

        params = CRIMES[crime]
        self.attempt_crime(
            s_k=params["min_sentence"],
            r_k=params["report_rate"],
            crime_name=crime,
            victim=victim,
        )
