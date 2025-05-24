import os
import networkx as nx

import matplotlib.pyplot as plt
import seaborn as sns

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
