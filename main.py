from model import CrimeSocietyModel
import visuals
import numpy as np
import pandas as pd


def main():
    num_steps = 60
    model = CrimeSocietyModel(wage_mode="A", num_agents=500, r_w=0.05)

    for _ in range(num_steps):
        model.step()

    final_df = model.snapshots[-1]

    for col in final_df.select_dtypes(include=[np.number]).columns:
        final_df[col] = final_df[col].round(2)

    final_df.to_csv("final_agents.csv", index=False)
    visuals.generate_all_plots(final_df, label="After Simulation")

    # Plot criminal network
    visuals.plot_criminal_network(model.agent_list, label="Criminal Network After Simulation")
    visuals.plot_criminal_only_network(model.agent_list, label="Criminal Association Network")
    
    visuals.plot_crime_histogram(model.crime_counts)

    initial_df = pd.read_csv("initial_agents.csv")
    visuals.plot_wealth_comparison(initial_df, final_df)

    visuals.plot_tracked_agent_life(model.tracked_life, model.tracked_id)

    # Load and compare wealth distributions
    initial_df = pd.read_csv("initial_agents.csv")
    visuals.report_gini(initial_df, final_df)

    visuals.plot_incarcerations(model.incarceration_log)

if __name__ == "__main__":
    main()
