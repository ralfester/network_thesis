import os
import networkx as nx

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

from enums import CriminalStatus
# --- Create plots directory ---
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# --- Plotting style ---
sns.set_theme(style="darkgrid")


def save_plot(fig, name):
    path = os.path.join(PLOT_DIR, f"{name}.pdf")
    fig.savefig(path)
    plt.close(fig)


def plot_hist(df, column, title, bins=30):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(df[column], kde=True, bins=bins, ax=ax)
    ax.set_title(title)
    save_plot(fig, f"{column}_hist")


def plot_count(df, column, title):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(data=df, x=column, ax=ax)
    ax.set_title(title)
    save_plot(fig, f"{column}_count")


def plot_top_nationalities(df, top_n=10):
    fig, ax = plt.subplots(figsize=(10, 6))
    top_nations = df["nationality"].value_counts().nlargest(top_n)
    sns.barplot(x=top_nations.index, y=top_nations.values, ax=ax)
    ax.set_title(f"Top {top_n} Nationalities")
    ax.tick_params(axis="x", rotation=45)
    save_plot(fig, "top_nationalities")


def generate_all_plots(df, label="snapshot"):
    plot_hist(df, "age", f"Age Distribution ({label})", bins=10)
    plot_count(df, "gender", f"Gender Distribution ({label})")
    plot_hist(df, "weight", f"Weight Distribution ({label})")
    plot_hist(df, "muscle_mass", f"Muscle Mass Distribution ({label})")
    plot_hist(df, "wealth", f"Wealth Distribution ({label})")
    plot_hist(df, "wage", f"Wage Distribution ({label})")
    plot_count(df, "criminal_status", f"Criminal Status Distribution ({label})")
    plot_top_nationalities(df)


def plot_criminal_network(agent_list, label="Criminal Network"):
    G = nx.Graph()

    # Add nodes with attributes
    for agent in agent_list:
        G.add_node(
            agent.unique_id,
            status=agent.criminal_status.name,
            wealth=agent.wealth,
            degree=len(agent.associates),
        )

    # Add edges from associations
    for agent in agent_list:
        for associate_id in agent.associates:
            if G.has_node(associate_id):
                G.add_edge(agent.unique_id, associate_id)

    # Create position layout
    pos = nx.spring_layout(G, seed=42)

    # Node color by criminal status
    status_colors = {
        "NON_CRIMINAL": "lightgray",
        "PETTY_CRIMINAL": "orange",
        "ORGANIZED_CRIMINAL": "red",
        "VORY": "purple",
    }
    node_colors = [status_colors[G.nodes[n]["status"]] for n in G.nodes]

    # Node size by degree
    node_sizes = [50 + 10 * G.nodes[n]["degree"] for n in G.nodes]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 9))
    nx.draw_networkx_nodes(
        G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8
    )
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    ax.set_title(label)
    ax.axis("off")

    save_plot(fig, "criminal_network")


print("visuals.py loaded. Use `generate_all_plots(df)` to create plots.")


def plot_full_network(agent_list, label="Full Social Network"):
    G = nx.Graph()

    for agent in agent_list:
        G.add_node(agent.unique_id, status=agent.criminal_status.name)
        for associate_id in agent.associates:
            G.add_edge(agent.unique_id, associate_id)

    pos = nx.spring_layout(G, seed=42)

    status_colors = {
        "NON_CRIMINAL": "lightgray",
        "PETTY_CRIMINAL": "orange",
        "ORGANIZED_CRIMINAL": "red",
        "VORY": "purple",
    }
    node_colors = [status_colors[G.nodes[n]["status"]] for n in G.nodes]
    node_sizes = [50 + 10 * G.degree[n] for n in G.nodes]

    fig, ax = plt.subplots(figsize=(12, 9))
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8
    )
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    ax.set_title(label)
    ax.axis("off")

    save_plot(fig, "full_network")


def plot_criminal_only_network(agent_list, label="Criminal Network Only"):
    G = nx.Graph()

    criminals = [
        a for a in agent_list if a.criminal_status != CriminalStatus.NON_CRIMINAL
    ]

    for agent in criminals:
        G.add_node(agent.unique_id, status=agent.criminal_status.name)
        for aid in agent.associates:
            associate = next((a for a in criminals if a.unique_id == aid), None)
            if associate:
                G.add_edge(agent.unique_id, aid)

    pos = nx.spring_layout(G, seed=24)

    status_colors = {
        "PETTY_CRIMINAL": "orange",
        "ORGANIZED_CRIMINAL": "red",
        "VORY": "purple",
    }
    node_colors = [status_colors[G.nodes[n]["status"]] for n in G.nodes]
    node_sizes = [60 + 12 * G.degree[n] for n in G.nodes]

    fig, ax = plt.subplots(figsize=(12, 9))
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9
    )
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    ax.set_title(label)
    ax.axis("off")

    save_plot(fig, "criminal_only_network")


def plot_crime_histogram(crime_counts, label="Crimes per Year", bin_size=12):
    bins = list(range(0, len(crime_counts) + bin_size, bin_size))
    yearly_crimes = [sum(crime_counts[i:i+bin_size]) for i in range(0, len(crime_counts), bin_size)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(yearly_crimes)), yearly_crimes)
    ax.set_xticks(range(len(yearly_crimes)))
    ax.set_xticklabels([f"Year {i+1}" for i in range(len(yearly_crimes))])
    ax.set_title(label)
    ax.set_ylabel("Total Crimes")
    ax.set_xlabel("Year")
    save_plot(fig, "crimes_per_year")


def plot_wealth_comparison(initial_df, final_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(initial_df["wealth"], label="Initial", ax=ax)
    sns.kdeplot(final_df["wealth"], label="Final", ax=ax)
    ax.set_title("Wealth Distribution: Start vs End")
    ax.set_xlabel("Wealth")
    ax.legend()
    save_plot(fig, "wealth_comparison")

def plot_tracked_agent_life(life_data, agent_id, label=None):
    df = pd.DataFrame(life_data)

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(df["step"], df["wealth"])
    axs[0].set_ylabel("Wealth")
    axs[0].set_title(label or f"Agent {agent_id} — Wealth Over Time")

    axs[1].bar(df["step"], df["crimes"])
    axs[1].set_ylabel("Crimes")
    axs[1].set_title("Crimes Per Step")
    axs[1].set_xlabel("Timestep")

    save_plot(fig, f"tracked_agent_{agent_id}_life")


def gini_coefficient(values):
    """
    Compute Gini coefficient of a numpy array of values.
    """
    array = np.array(values)
    if np.amin(array) < 0:
        array -= np.amin(array)
    array += 1e-8  # avoid division by zero
    array = np.sort(array)
    n = len(array)
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))


def report_gini(initial_df, final_df):
    gini_start = gini_coefficient(initial_df["wealth"])
    gini_end = gini_coefficient(final_df["wealth"])

    print(f"Initial Gini Coefficient: {gini_start:.4f}")
    print(f"Final Gini Coefficient:   {gini_end:.4f}")


def plot_incarcerations(incarceration_log, bin_size=12, label="Incarcerations per Year"):
    yearly = [sum(incarceration_log[i:i+bin_size]) for i in range(0, len(incarceration_log), bin_size)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(yearly)), yearly, color="steelblue")
    ax.set_xticks(range(len(yearly)))
    ax.set_xticklabels([f"Year {i+1}" for i in range(len(yearly))])
    ax.set_ylabel("People Incarcerated")
    ax.set_xlabel("Year")
    ax.set_title(label)
    save_plot(fig, "incarcerations_per_year")