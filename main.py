from model import CrimeSocietyModel
import visuals
import numpy as np


def main():
    num_steps = 36
    model = CrimeSocietyModel(wage_mode="A", num_agents=100, r_w=0.05)

    for _ in range(num_steps):
        model.step()

    final_df = model.snapshots[-1]

    for col in final_df.select_dtypes(include=[np.number]).columns:
        final_df[col] = final_df[col].round(2)

    final_df.to_csv("final_agents.csv", index=False)
    visuals.generate_all_plots(final_df, label="After Simulation")

    # Plot criminal network
    visuals.plot_criminal_network(model.agent_list, label="Criminal Network After Simulation")


if __name__ == "__main__":
    main()
