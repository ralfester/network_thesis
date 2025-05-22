import os
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


print("visuals.py loaded. Use `generate_all_plots(df)` to create plots.")
