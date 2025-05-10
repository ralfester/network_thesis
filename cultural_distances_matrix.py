import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

df = pd.read_csv("download_matrix2.csv")
df = df.fillna(0)
df = df.map(lambda x: f"{x:.3g}" if isinstance(x, (int, float)) else x)
#  print(df.head())

df.columns = df.columns.str.replace(r"\d{4}-\d{4}", "", regex=True)

if df.shape[0] == df.shape[1]:
    df.insert(0, " ", df.columns)
else:
    raise ValueError(
        "DataFrame is not square — cannot align row index with column names."
    )

with PdfPages("testpdf.pdf") as pdf:
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.axis("off")

    fig.suptitle(
        "Cultural Distance of Nations 1981–2014",
        fontsize=16,
        fontweight="bold",
        color="#1B2631",
        y=0.95,
    )

    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center")

    table.auto_set_font_size(True)
    table.scale(1.2, 1.2)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2E86C1")  # Header row color (blue)
            cell.set_text_props(color="white", weight="bold")
        if col == 0 and row != 0:
            cell.set_facecolor("#AED6F1")  # First column color (light blue)
            cell.set_text_props(weight="bold")

    pdf.savefig()
