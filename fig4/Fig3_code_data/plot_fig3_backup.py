from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)
DATA_FILE = DATA_DIR / "attention_with_attention_and_signal_filtered.fasta"

# =========================
# Plot style
# =========================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11.8,
    "axes.labelsize": 10.2,
    "xtick.labelsize": 9.1,
    "ytick.labelsize": 9.3,
    "legend.fontsize": 8.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "grid.color": "#D9DEE6",
    "grid.linewidth": 0.75,
    "grid.alpha": 0.65,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

GREEN = "#1E7F35"
GREEN_DARK = "#145A26"
GREY = "#D8D8D8"
GREY_EDGE = "#B8B8B8"
TEAL = "#287E94"   # panel C non-black color
TEXT = "#111111"
SUBTLE = "#6E6E6E"


# =========================
# Data helpers
# =========================
def parse_fasta(path):
    entries = []
    with open(path, "r", encoding="utf-8") as fh:
        lines = [line.rstrip("\n") for line in fh if line.strip()]
    i = 0
    while i < len(lines):
        if lines[i].startswith(">") and i + 3 < len(lines):
            entries.append({
                "id": lines[i][1:],
                "seq": lines[i + 1],
                "model": lines[i + 2],
                "standard": lines[i + 3],
            })
            i += 4
        else:
            i += 1
    return entries


def match_window(pos, standard, window=1):
    n = len(standard)
    return any(0 <= pos + d < n and standard[pos + d] == "S" for d in range(-window, window + 1))


def match_exact(pos, standard):
    return pos < len(standard) and standard[pos] == "S"


def method_stats(entries, matcher):
    seq_matched = 0
    matched_positions = 0
    unmatched_positions = 0
    first_positions = []

    for e in entries:
        model = e["model"]
        standard = e["standard"]
        detected = False
        for i, c in enumerate(model):
            if c != "S":
                continue
            is_match = matcher(i, standard)
            if is_match:
                matched_positions += 1
                if not detected:
                    first_positions.append(i + 1)
                    detected = True
            else:
                unmatched_positions += 1
        if detected:
            seq_matched += 1

    return {
        "seq_matched": seq_matched,
        "seq_unmatched": len(entries) - seq_matched,
        "pos_matched": matched_positions,
        "pos_unmatched": unmatched_positions,
        "first_positions": sorted(first_positions),
    }


def motif_counts(entries, matcher, k=7, top_n=10):
    half = k // 2
    kmers = Counter()
    for e in entries:
        seq = e["seq"]
        model = e["model"]
        standard = e["standard"]
        n = min(len(seq), len(model), len(standard))
        for i in range(n):
            if model[i] == "S" and not matcher(i, standard):
                start, end = i - half, i + half + 1
                if start >= 0 and end <= n:
                    kmers[seq[start:end]] += 1
    return kmers.most_common(top_n)


entries = parse_fasta(DATA_FILE)
n_total = len(entries)

# Customer-specified final display: remove blue method everywhere,
# keep only Method 2 (point-to-point), and split green/grey into separate bars.
method_name = "Point-to-point (Method 2)"
stats = method_stats(entries, match_exact)
# Keep original motif counting basis used previously for panel C ranking.
top_motifs = motif_counts(entries, match_window, k=7, top_n=10)

# =========================
# Figure
# =========================
fig = plt.figure(figsize=(13.8, 8.9))
gs = GridSpec(2, 2, figure=fig, hspace=0.46, wspace=0.24)

# ---------- a: sequence-level detection rate ----------
ax_a = fig.add_subplot(gs[0, 0])
seq_labels = ["Detected", "Not detected"]
seq_counts = [stats["seq_matched"], stats["seq_unmatched"]]
seq_values = [v / n_total * 100 for v in seq_counts]
seq_colors = [GREEN, GREY]
seq_edges = ["white", GREY_EDGE]
ypos = np.array([1, 0])

ax_a.barh(ypos, seq_values, color=seq_colors, edgecolor=seq_edges, linewidth=0.9, height=0.44, zorder=3)
for y, label, count, val, color in zip(ypos, seq_labels, seq_counts, seq_values, seq_colors):
    txt = f"{count:,} ({val:.1f}%)"
    if label == "Detected":
        ax_a.text(val / 2, y, txt, ha="center", va="center", color="white", fontsize=10.4, fontweight="bold",
                  path_effects=[pe.withStroke(linewidth=1.4, foreground="black", alpha=0.14)])
    else:
        ax_a.text(min(val + 2.2, 101.5), y, txt, ha="left", va="center", color=TEXT, fontsize=10.1, fontweight="bold")

ax_a.set_title("Sequence-level detection rate", fontweight="bold", pad=12)
ax_a.text(0.00, 1.07, method_name, transform=ax_a.transAxes, ha="left", va="bottom",
          fontsize=10.1, color=GREEN_DARK, fontweight="bold")
ax_a.text(1.00, 1.07, f"n = {n_total:,} sequences", transform=ax_a.transAxes, ha="right", va="bottom",
          fontsize=10.0, color=TEXT)
ax_a.set_xlim(0, 105)
ax_a.set_xlabel("Percentage of sequences")
ax_a.set_yticks(ypos)
ax_a.set_yticklabels(seq_labels)
for tick, lbl in zip(ax_a.get_yticklabels(), seq_labels):
    tick.set_fontweight("bold")
    tick.set_color(GREEN_DARK if lbl == "Detected" else SUBTLE)
ax_a.grid(axis="x", zorder=0)
ax_a.text(-0.13, 1.10, "a", transform=ax_a.transAxes, fontsize=22, fontweight="bold", va="bottom")

# ---------- b: position-level agreement ----------
ax_b = fig.add_subplot(gs[0, 1])
pos_labels = ["Matched within criterion", "Unmatched"]
pos_counts = [stats["pos_matched"], stats["pos_unmatched"]]
pos_total = sum(pos_counts)
pos_values = [v / pos_total * 100 for v in pos_counts]
pos_colors = [GREEN, GREY]
pos_edges = ["white", GREY_EDGE]
ypos_b = np.array([1, 0])

ax_b.barh(ypos_b, pos_values, color=pos_colors, edgecolor=pos_edges, linewidth=0.9, height=0.44, zorder=3)
for y, label, count, val in zip(ypos_b, pos_labels, pos_counts, pos_values):
    txt = f"{count:,} ({val:.1f}%)"
    if label.startswith("Matched"):
        ax_b.text(val / 2, y, txt, ha="center", va="center", color="white", fontsize=10.1, fontweight="bold",
                  path_effects=[pe.withStroke(linewidth=1.4, foreground="black", alpha=0.14)])
    else:
        ax_b.text(val / 2, y, txt, ha="center", va="center", color=TEXT, fontsize=10.0, fontweight="bold")

ax_b.set_title("Position-level agreement of high-attention sites", fontweight="bold", pad=12)
ax_b.text(0.00, 1.07, method_name, transform=ax_b.transAxes, ha="left", va="bottom",
          fontsize=10.1, color=GREEN_DARK, fontweight="bold")
ax_b.text(1.00, 1.07, f"Total high-attention positions = {pos_total:,}", transform=ax_b.transAxes,
          ha="right", va="bottom", fontsize=10.0, color=TEXT)
ax_b.set_xlim(0, 105)
ax_b.set_xticks([0, 25, 50, 75, 100])
ax_b.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
ax_b.set_xlabel("Percentage of high-attention positions")
ax_b.set_yticks(ypos_b)
ax_b.set_yticklabels(pos_labels)
for tick, lbl in zip(ax_b.get_yticklabels(), pos_labels):
    tick.set_fontweight("bold")
    tick.set_color(GREEN_DARK if lbl.startswith("Matched") else SUBTLE)
ax_b.grid(axis="x", zorder=0)
legend_handles = [
    mpatches.Patch(color=GREEN, label="Matched within criterion"),
    mpatches.Patch(color=GREY, label="Unmatched"),
]
ax_b.legend(handles=legend_handles, frameon=False, loc="upper left", bbox_to_anchor=(0.00, 1.00),
            ncol=2, columnspacing=1.0, handlelength=1.1)
ax_b.text(-0.13, 1.10, "b", transform=ax_b.transAxes, fontsize=22, fontweight="bold", va="bottom")

# ---------- c: motif lollipop panel ----------
ax_c = fig.add_subplot(gs[1, 0])
motifs, counts = zip(*top_motifs)
motifs = list(motifs)
counts = np.array(counts)
y = np.arange(len(motifs))[::-1]

ax_c.hlines(y, 0, counts, color=TEAL, lw=1.45, alpha=0.95, zorder=2)
ax_c.scatter(counts, y, s=46, color=TEAL, edgecolor="#1A5F71", linewidth=0.6, zorder=4)
for rank, yi in enumerate(y, start=1):
    ax_c.scatter(-4.8, yi, s=240, color="#D9EEF3", edgecolor="white", linewidth=0.8, zorder=3, clip_on=False)
    ax_c.text(-4.8, yi, str(rank), ha="center", va="center", color=TEXT, fontsize=9.1, fontweight="bold", clip_on=False)
for yi, motif, count in zip(y, motifs, counts):
    ax_c.text(-3.6, yi, motif, ha="left", va="center", color="#0B5F73", fontsize=10.0, fontweight="bold", family="DejaVu Sans Mono")
    ax_c.text(16.0, yi, f"{int(count)}", ha="right", va="center", color="#0B5F73", fontsize=10.0, fontweight="bold")

ax_c.set_xlim(-6.5, 16.4)
ax_c.set_ylim(-0.8, len(motifs) - 0.2)
ax_c.set_yticks([])
ax_c.set_xlabel("Count", color="#0B5F73")
ax_c.tick_params(axis="x", colors="#0B5F73")
ax_c.spines["left"].set_visible(False)
ax_c.spines["bottom"].set_color("#0B5F73")
ax_c.grid(axis="x", zorder=0)
ax_c.set_title("Top recurring 7-mer motifs in non-SP high-attention regions", loc="left", fontweight="bold", pad=18)
ax_c.text(0.00, 1.015, "Ranking shared by both matching criteria", transform=ax_c.transAxes,
          ha="left", va="bottom", fontsize=9.8, style="italic", color="#0B5F73")
ax_c.text(0.00, 0.972, "Motif", transform=ax_c.transAxes, ha="left", va="center", fontsize=9.8, color="#0B5F73")
ax_c.text(0.98, 0.972, "Count", transform=ax_c.transAxes, ha="right", va="center", fontsize=9.8, color="#0B5F73", fontweight="bold")
ax_c.text(-0.13, 1.10, "c", transform=ax_c.transAxes, fontsize=22, fontweight="bold", va="bottom")

# ---------- d: cumulative first detection ----------
ax_d = fig.add_subplot(gs[1, 1])
fp = np.array(stats["first_positions"], dtype=float)
cum = np.arange(1, len(fp) + 1) / n_total * 100
ax_d.step(fp, cum, where="post", color=GREEN, lw=2.15, label=method_name, zorder=3)
ax_d.fill_between(fp, cum, step="post", color=GREEN, alpha=0.09, zorder=1)
med = int(np.median(fp)) if len(fp) else 0
if med:
    ax_d.axvline(med, color=SUBTLE, linestyle="--", lw=1.0, alpha=0.8, zorder=2)
    ax_d.text(med + 0.35, 6, f"Median: {med} aa", color=SUBTLE, fontsize=9.2, style="italic")
ax_d.set_xlim(1, 20.2)
ax_d.set_ylim(0, 102)
ax_d.set_title("Cumulative first matched detection position", fontweight="bold", pad=12)
ax_d.text(0.00, 1.07, method_name, transform=ax_d.transAxes, ha="left", va="bottom",
          fontsize=10.0, color=GREEN_DARK, fontweight="bold")
ax_d.set_xlabel("First matched detection position (aa, 1-based)")
ax_d.set_ylabel("Cumulative % of all sequences")
ax_d.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
ax_d.grid(axis="both", zorder=0)
# no legend needed because only one method remains
ax_d.text(-0.13, 1.10, "d", transform=ax_d.transAxes, fontsize=22, fontweight="bold", va="bottom")

fig.subplots_adjust(left=0.085, right=0.985, top=0.94, bottom=0.09)
for ext in ["png", "svg"]:
    fig.savefig(OUT_DIR / f"Fig3_scientific_composite.{ext}", dpi=320 if ext == "png" else None,
                bbox_inches="tight", facecolor="white")
plt.close(fig)

with open(OUT_DIR / "VERIFICATION.txt", "w", encoding="utf-8") as fh:
    fh.write("Fig3 customer-request final version\n")
    fh.write("Rules applied:\n")
    fh.write("1. Blue method removed from all panels.\n")
    fh.write("2. Panels a and b split green and grey into separate bars.\n")
    fh.write("3. No underlying data changed.\n\n")
    fh.write(f"Loaded entries: {n_total}\n")
    fh.write(f"Method retained: {method_name}\n")
    fh.write(f"Sequence detected: {stats['seq_matched']}/{n_total}\n")
    fh.write(f"Sequence not detected: {stats['seq_unmatched']}/{n_total}\n")
    fh.write(f"Position matched: {stats['pos_matched']}/{pos_total}\n")
    fh.write(f"Position unmatched: {stats['pos_unmatched']}/{pos_total}\n")
    fh.write(f"Median first matched detection position: {med}\n")

print(f"Loaded entries: {n_total}")
print(f"Sequence detected: {stats['seq_matched']}/{n_total}")
print(f"Position matched: {stats['pos_matched']}/{pos_total}")
print(f"Saved outputs to: {OUT_DIR}")
