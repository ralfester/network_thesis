import numpy as np

age_groups = np.array(
    [
        (15, 19, 6.87),
        (20, 24, 6.61),
        (25, 29, 8.49),
        (30, 34, 8.75),
        (35, 39, 8.02),
        (40, 44, 5.58),
        (45, 49, 4.96),
        (50, 54, 6.69),
        (55, 59, 5.53),
        (60, 64, 5.76),
        (65, 69, 3.21),
    ]
)


group_starts = age_groups[:, 0].astype(int)
group_ends = age_groups[:, 1].astype(int)
group_weights = age_groups[:, 2]

group_probs = group_weights / group_weights.sum()


def sample_ages(n_samples):

    chosen_indices = np.random.choice(len(age_groups),
                                      size=n_samples, p=group_probs)

    sampled_ages = np.array(
        [np.random.randint(group_starts[i], group_ends[i] + 1) for i in chosen_indices]
    )

    return sampled_ages


# Example usage
samples = sample_ages(10)
print(samples)
