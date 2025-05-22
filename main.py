from model import CrimeSocietyModel
import visuals
import numpy as np


def main():
    num_steps = 10
    model = CrimeSocietyModel(num_agents=1000)

    for _ in range(num_steps):
        model.step()

    final_df = model.snapshots[-1]

    for col in final_df.select_dtypes(include=[np.number]).columns:
        final_df[col] = final_df[col].round(2)

    final_df.to_csv("final_agents.csv", index=False)
    visuals.generate_all_plots(final_df, label="After Simulation")


if __name__ == "__main__":
    main()
