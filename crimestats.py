import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Load and process the data
df = pd.read_csv("crimestats.csv", sep=';')
df['number of offences'] = pd.to_numeric(df['number of offences'], errors='coerce')
df['offences per 100000'] = (df['number of offences'] / 1370052) * 100000
df['offences per 100000'] = df['offences per 100000'].apply(lambda x: float(f"{x:.3g}") if pd.notnull(x) else np.nan)

# Select relevant columns
table_data = df[['Crime', 'offences per 100000']]

# Create PDF
with PdfPages('crime_rates.pdf') as pdf:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')  # No axes

    # Create the table
    table = ax.table(cellText=table_data.values,
                     colLabels=table_data.columns,
                     loc='center',
                     cellLoc='left',
                     colLoc='left')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
