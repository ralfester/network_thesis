import mesa
import numpy as np
import random
import math


class PersonAgent(mesa.Agent):
    def __init__(self, unique_id, model):  # initialize agents with unique ID
        super().__init__(unique_id, model)

        # Demographics of agents
        self.gender = self.random.choices(["F", "M"], weights=[0.54, 0.46])[0]
        self.age = self.sample_age()  # samples age from our age distribution
        self.height = self.random.gauss(mu=170, sigma=10)  # decorative
        self.weight = self.sample_weight_log_normal(self.gender)  # decorative

        # Muscle mass modeling
        self.muscle_mass_initial = self.sample_muscle_mass_log_normal(self.gender)
        self.muscle_mass = self.compute_muscle_mass(self.age, self.muscle_mass_initial)

        # Social trait, needs to be added to text
        self.charisma = self.random.uniform(0, 1)

        # Crime & life state, starts as false
        self.is_criminal = False
        self.criminal_status = None
        self.incarcerated = False
        self.desisted = False

        # Spatial location, taken from our equation
        self.location = self.assign_location()

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
        return random.randint(*selected_bin)  # this way, we get set age

    def sample_weight_log_normal(self, gender):
        if gender == "M":
            mu, sigma = 4.25, 0.17  # ln(70kg median), approx SD
        else:
            mu, sigma = 4.04, 0.20  # ln(57kg median), approx SD
        return np.random.lognormal(mean=mu, sigma=sigma)

    def sample_muscle_mass_log_normal(self, gender):
        if gender == "M":
            mu, sigma = 3.5, 0.12  # ln(33kg median), approx SD
        else:
            mu, sigma = 3.09, 0.15  # ln(22kg median), approx SD
        return np.random.lognormal(mean=mu, sigma=sigma)

    def compute_muscle_mass(self, age, initial_mass):
        """
        Apply age-based exponential decay to muscle mass based on:
        - No loss before 30
        - 1% per year between 30–50
        - 1.5% per year after 50
        """
        if age < 30:
            return initial_mass
        elif age < 50:
            decay_years = age - 30
            return initial_mass * math.exp(-0.01 * decay_years)
        else:
            # compound decay: 20 years at 1%, rest at 1.5%
            decay_years_1 = 20
            decay_years_2 = age - 50
            intermediate_mass = initial_mass * math.exp(-0.01 * decay_years_1)
            return intermediate_mass * math.exp(-0.015 * decay_years_2)

    def assign_location(self):
        # Bias toward central ring (radius ≤ 3km)
        angle = self.random.uniform(0, 2 * np.pi)
        radius = np.sqrt(self.random.uniform(0, 1)) * 3
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        return (x, y)
