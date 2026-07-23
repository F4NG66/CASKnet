from pathlib import Path
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "grid.color": "#D7DCE2",
    "grid.linewidth": 0.75,
    "grid.alpha": 0.65,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

COLORS = {
    "neg_val": "#183E68",
    "pos_val": "#4E908B",
    "neg_cur": "#D65A42",
    "grey_before": "#C9C9C9",
    "grey_after": "#4F4F4F",
    "archaea": "#8F4B99",
}

# These CSV files are derived directly from the raw files included in data/.
# No data are modified in this script; the script only reads the supplied files and draws the figure.
length_df = pd.read_csv(DATA_DIR / "fig1_length_distribution.csv")
dedup_df = pd.read_csv(DATA_DIR / "fig1_dedup_counts.csv")
tax_df = pd.read_csv(DATA_DIR / "fig1_taxonomy_counts.csv")

bin_labels = length_df["bin"].astype(str).tolist()
length_pivot = length_df.set_index("bin")[["neg_val", "pos_val", "neg_cur"]]

order_cats = ["pos_val", "neg_val", "neg_cur"]
cat_labels = {"pos_val": "Pos. Validated", "neg_val": "Neg. Validated", "neg_cur": "Neg. Curated"}

tax_order = ["Eukaryota", "Bacteria", "Viruses", "Archaea"]
tax_colors = [COLORS["neg_val"], COLORS["pos_val"], COLORS["neg_cur"], COLORS["archaea"]]

fig = plt.figure(figsize=(13.5, 8.2), constrained_layout=False)
gs = GridSpec(
    2, 2, figure=fig,
    height_ratios=[1.05, 0.85], width_ratios=[1.03, 1.0],
    hspace=0.48, wspace=0.22
)

# Panel a
ax_a = fig.add_subplot(gs[0, 0])
x = np.arange(len(bin_labels))
bottom = np.zeros(len(bin_labels))
for key, label, color in [
    ("neg_val", "Neg. Validated", COLORS["neg_val"]),
    ("pos_val", "Pos. Validated", COLORS["pos_val"]),
    ("neg_cur", "Neg. Curated", COLORS["neg_cur"]),
]:
    vals = length_pivot[key].values
    ax_a.bar(x, vals, width=0.66, bottom=bottom, color=color, edgecolor="white", linewidth=0.8, label=label, zorder=3)
    for xi, v, btm in zip(x, vals, bottom):
        if v > 0:
            ax_a.text(
                xi, btm + v / 2, f"{int(v):,}",
                ha="center", va="center", color="white",
                fontsize=9.5, fontweight="bold",
                path_effects=[pe.withStroke(linewidth=1.4, foreground="black", alpha=0.2)]
            )
    bottom += vals

ax_a.set_title("Sequence length distribution by sample category", pad=10, fontweight="bold")
ax_a.set_xlabel("Sequence length (aa)", fontweight="bold")
ax_a.set_ylabel("Count", fontweight="bold")
ax_a.set_xticks(x)
ax_a.set_xticklabels(bin_labels)
ax_a.set_ylim(0, max(bottom) * 1.18)
ax_a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax_a.grid(axis="y", zorder=0)
ax_a.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.98, 1.02), handlelength=1.2)
ax_a.text(-0.13, 1.07, "a", transform=ax_a.transAxes, fontsize=22, fontweight="bold", va="bottom")

# Panel b
ax_b = fig.add_subplot(gs[0, 1])
dedup_map = dedup_df.set_index("category")
orig = dedup_map.loc[order_cats, "original"].values
dedup = dedup_map.loc[order_cats, "deduplicated"].values
x2 = np.arange(len(order_cats))
w = 0.32
ax_b.bar(x2 - w / 2, orig, width=w, color=COLORS["grey_before"], edgecolor="white", linewidth=0.9, label="Before deduplication", zorder=3)
ax_b.bar(x2 + w / 2, dedup, width=w, color=COLORS["grey_after"], edgecolor="white", linewidth=0.9, label="After deduplication", zorder=3)

for i, (o, d) in enumerate(zip(orig, dedup)):
    ax_b.text(i - w / 2, o + 110, f"{int(o):,}", ha="center", va="bottom", fontsize=9.2, fontweight="bold")
    ax_b.text(i + w / 2, d + 110, f"{int(d):,}", ha="center", va="bottom", fontsize=9.2, fontweight="bold")
    ax_b.text(
        i + w / 2, d + 520, f"{d / o * 100:.0f}%",
        ha="center", va="bottom", color="#1C66D4", fontsize=10.5, fontweight="bold"
    )

ax_b.set_title("Sample counts before and after deduplication", pad=10, fontweight="bold")
ax_b.set_ylabel("Number of sequences", fontweight="bold")
ax_b.set_xticks(x2)
ax_b.set_xticklabels([cat_labels[c] for c in order_cats])
ax_b.set_ylim(0, max(orig.max(), dedup.max()) * 1.25)
ax_b.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax_b.grid(axis="y", zorder=0)
ax_b.legend(frameon=False, loc="upper right")
ax_b.text(-0.13, 1.07, "b", transform=ax_b.transAxes, fontsize=22, fontweight="bold", va="bottom")

# Panel c
ax_c = fig.add_subplot(gs[1, :])
row_y = np.arange(len(tax_df))[::-1]
for row_idx, (_, row) in enumerate(tax_df.iterrows()):
    y = row_y[row_idx]
    vals = np.array([row[t] for t in tax_order], dtype=float)
    total = float(row["total"])
    props = vals / total * 100 if total else np.zeros_like(vals)
    left = 0.0
    for prop, val, color in zip(props, vals, tax_colors):
        ax_c.barh(y, prop, left=left, height=0.62, color=color, edgecolor="white", linewidth=1.0, zorder=3)
        if prop >= 3.0:
            ax_c.text(
                left + prop / 2, y, f"{prop:.1f}%",
                ha="center", va="center", color="white", fontsize=10, fontweight="bold",
                path_effects=[pe.withStroke(linewidth=1.2, foreground="black", alpha=0.18)]
            )
        elif prop > 0:
            ax_c.plot([left + prop, left + prop + 1.8], [y, y + 0.04], color=color, lw=0.8)
            ax_c.text(left + prop + 2.2, y + 0.04, f"{prop:.1f}%", ha="left", va="center", color=color, fontsize=9.2, fontweight="bold")
        left += prop

# Use explicit text positions instead of overlapping title/subtitle.
# This only changes spacing; no data or plotted values are changed.
ax_c.text(0.0, 1.135, "Domain / realm proportions", transform=ax_c.transAxes,
          fontsize=10, style="italic", color="#333333", ha="left", va="bottom")
ax_c.text(0.0, 1.045, "Taxonomic composition", transform=ax_c.transAxes,
          fontsize=12, fontweight="bold", color="black", ha="left", va="bottom")
ax_c.set_yticks(row_y)
ax_c.set_yticklabels([f"{row['sample_group']}\n(n={int(row['total']):,})" for _, row in tax_df.iterrows()])
ax_c.set_xlim(0, 104)
ax_c.set_xlabel("Proportion of sequences (%)", fontweight="bold", labelpad=8)
ax_c.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
ax_c.set_xticks(np.arange(0, 101, 20))
ax_c.grid(axis="x", zorder=0)
ax_c.spines["left"].set_visible(False)
ax_c.tick_params(axis="y", length=0)
ax_c.legend(
    handles=[mpatches.Patch(color=c, label=t) for t, c in zip(tax_order, tax_colors)],
    loc="upper center", bbox_to_anchor=(0.5, -0.27), frameon=False,
    ncol=4, columnspacing=2.6, handlelength=1.4
)
ax_c.text(-0.11, 1.205, "c", transform=ax_c.transAxes, fontsize=22, fontweight="bold", va="bottom")

fig.subplots_adjust(left=0.08, right=0.985, top=0.94, bottom=0.15)

png_path = OUT_DIR / "Fig1_scientific_composite.png"
vector_svg_path = OUT_DIR / "Fig1_scientific_composite_vector.svg"
svg_path = OUT_DIR / "Fig1_scientific_composite.svg"

fig.savefig(png_path, dpi=320, bbox_inches="tight", facecolor="white")
fig.savefig(vector_svg_path, bbox_inches="tight", facecolor="white")
plt.close(fig)

# Create an SVG file that displays the same PNG exactly, so the SVG appearance stays
# identical to the checked PNG in all viewers.
try:
    from PIL import Image
    img = Image.open(png_path)
    width, height = img.size
except Exception:
    width, height = 4320, 2624

with open(png_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("ascii")
svg_text = (
    f'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
    f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
    f'width="{width}" height="{height}" viewBox="0 0 {width} {height}" version="1.1">\n'
    f'  <image width="{width}" height="{height}" x="0" y="0" '
    f'href="data:image/png;base64,{b64}" xlink:href="data:image/png;base64,{b64}" />\n'
    f'</svg>\n'
)
with open(svg_path, "w", encoding="utf-8") as f:
    f.write(svg_text)

print(f"Saved PNG: {png_path}")
print(f"Saved SVG matching PNG: {svg_path}")
print(f"Saved optional vector SVG: {vector_svg_path}")
