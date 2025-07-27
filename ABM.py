import mesa
import numpy as np
import random
from enum import Enum, auto
from math import exp, sqrt

# Cultural module import
from Culture_language_mechanism import (
    pick_nationality,
    get_languages,
    cultural_favorability,
)


# CRIMINAL STATUS ENUM
class CriminalStatus(Enum):
    NON_CRIMINAL = auto()
    PETTY_CRIMINAL = auto()
    ORGANIZED_CRIMINAL = auto()
    VORY = auto()


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
