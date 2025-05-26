import numpy as np

# Placeholder for post-Soviet states with placeholder probabilities
NATIONALITIES = {
    "Russia": 0.52,
    "Ukraine": 0.15,
    "Belarus": 0.05,
    "Uzbekistan": 0.06,
    "Kazakhstan": 0.05,
    "Georgia": 0.03,
    "Armenia": 0.02,
    "Azerbaijan": 0.02,
    "Moldova": 0.02,
    "Latvia": 0.01,
    "Lithuania": 0.01,
    "Estonia": 0.01,
    "Kyrgyzstan": 0.02,
    "Tajikistan": 0.02,
    "Turkmenistan": 0.01,
}

LANGUAGE_PROFILES = {
    "Russia": ("Russian", None),
    "Ukraine": ("Ukrainian", "Russian"),
    "Belarus": ("Belarusian", "Russian"),
    "Kazakhstan": ("Kazakh", "Russian"),
    "Uzbekistan": ("Uzbek", "Russian"),
    "Latvia": ("Latvian", "Russian"),
    "Lithuania": ("Lithuanian", "Russian"),
    # Will extend later
}

# Cultural distance (must be symmetric matrix: placeholder values for now)
CULTURAL_DISTANCES = {
    ("Russia", "Russia"): 0.0,
    ("Russia", "Ukraine"): 0.1,
    ("Russia", "Kazakhstan"): 0.2,
    ("Ukraine", "Georgia"): 0.3,
    # TO DO: complete pairwise entries
}


def pick_nationality():
    countries = list(NATIONALITIES.keys())
    probs = list(NATIONALITIES.values())
    return np.random.choice(countries, p=probs)


def get_languages(nationality):
    return LANGUAGE_PROFILES.get(nationality, (nationality, None))


def cultural_favorability(nat1, nat2):
    key = (nat1, nat2) if (nat1, nat2) in CULTURAL_DISTANCES else (nat2, nat1)
    cd = CULTURAL_DISTANCES.get(key, 1.0)  # maximum distance if unknown
    return 1 - cd


class Agent:
    def __init__(self, id):
        self.id = id
        self.nationality = pick_nationality()
        self.language_primary, self.language_secondary = get_languages(self.nationality)

    def speaks(self, lang):
        return lang == self.language_primary or lang == self.language_secondary

    def interaction_favorability(self, other_agent):
        return cultural_favorability(self.nationality, other_agent.nationality)
