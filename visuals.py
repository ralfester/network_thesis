import os
import matplotlib.pyplot as plt
import seaborn as sns
from model import CrimeSocietyModel

# --- Run simulation to get initialized agent data ---
model = CrimeSocietyModel(num_agents=1000)
initial_df = model.agent_dataframe.copy()

# --- Export to CSV ---
initial_df.to_csv("initial_agents.csv", index=False)
print("Initial agent data exported to 'initial_agents.csv'.")

# --- Create plots directory ---
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# --- Plotting style ---
sns.set_theme(style="darkgrid")


# --- Plot functions ---
def save_plot(fig, name):
    path = os.path.join(PLOT_DIR, f"{name}.pdf")
    fig.savefig(path)
    plt.close(fig)


def plot_hist(column, title, bins=30):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(initial_df[column], kde=True, bins=bins, ax=ax)
    ax.set_title(title)
    save_plot(fig, column)


def plot_count(column, title):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(data=initial_df, x=column, ax=ax)
    ax.set_title(title)
    save_plot(fig, column)


# --- Generate and save plots ---
plot_hist("age", "Age Distribution", bins=10)
plot_count("gender", "Gender Distribution")
plot_hist("weight", "Weight Distribution")
plot_hist("muscle_mass", "Muscle Mass Distribution")
plot_hist("wealth", "Initial Wealth Distribution")
plot_hist("wage", "Initial Wage Distribution")
plot_count("criminal_status", "Criminal Status Distribution")

# --- Top Nationalities ---
fig, ax = plt.subplots(figsize=(10, 6))
top_nations = initial_df["nationality"].value_counts().nlargest(10)
sns.barplot(x=top_nations.index, y=top_nations.values, ax=ax)
ax.set_title("Top 10 Nationalities")
ax.tick_params(axis="x", rotation=45)
save_plot(fig, "top_nationalities")

print(f"All plots saved to ./{PLOT_DIR}/")
