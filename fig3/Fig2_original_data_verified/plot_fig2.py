from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    "xtick.labelsize": 9.3,
    "ytick.labelsize": 9.3,
    "legend.fontsize": 9.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "grid.color": "#D9DEE6",
    "grid.linewidth": 0.75,
    "grid.alpha": 0.70,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

COLORS = {
    "CASKnet": "#073B85",
    "SignalP6": "#0E8C7C",
    "TargetP2": "#E59B00",
    "CNN": "#0E8C7C",
    "ProtT5": "#D4B100",
    "Transformer": "#7652B5",
}
LINESTYLES = {
    "CASKnet": "-",
    "CNN": "-",
    "ProtT5": "--",
    "Transformer": ":",
}
MARKERS = {
    "CASKnet": "o",
    "CNN": "o",
    "ProtT5": "s",
    "Transformer": "^",
}
SUBTLE = "#6E6E6E"


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def parse_mean_sd(value):
    s = str(value).strip()
    m = re.match(r"([-\d.]+)\s*±\s*([\d.]+)", s)
    if m:
        return float(m.group(1)), float(m.group(2))
    return float(s), 0.0


def midpoint(label):
    a, b = label.split("-")
    return (int(a) + int(b)) / 2


gt = pd.read_csv(DATA_DIR / "final_deduplicated_dataset.csv")
cask_raw = pd.read_csv(DATA_DIR / "casknet_predictions.csv")
tp2_raw = pd.read_csv(DATA_DIR / "targetp2.csv")
sp6_raw = pd.read_csv(DATA_DIR / "signalp6.csv")

tp2_raw["pred_bin"] = (tp2_raw["Prediction"] == "SP").astype(int)
sp6_raw["pred_bin"] = (sp6_raw["Prediction"] == "SP").astype(int)
cask_raw["pred_bin"] = cask_raw["Prediction"].astype(int)

gt_slim = gt[["Entry", "Sequence", "label"]].rename(columns={"Entry": "ID"})
gt_label_only = gt[["Entry", "label"]].rename(columns={"Entry": "ID"})

cask = cask_raw.merge(gt_slim, on="ID", how="inner", suffixes=("", "_gt"))
tp2 = tp2_raw.merge(gt_label_only, on="ID", how="inner")
sp6 = sp6_raw.merge(gt_label_only, on="ID", how="inner")

common_ids = set(cask["ID"]) & set(tp2["ID"]) & set(sp6["ID"])
cask = cask[cask["ID"].isin(common_ids)].sort_values("ID").reset_index(drop=True)
tp2 = tp2[tp2["ID"].isin(common_ids)].sort_values("ID").reset_index(drop=True)
sp6 = sp6[sp6["ID"].isin(common_ids)].sort_values("ID").reset_index(drop=True)

y_true = cask["label"].to_numpy()
preds = {
    "CASKnet": cask["pred_bin"].to_numpy(),
    "SignalP6": sp6["pred_bin"].to_numpy(),
    "TargetP2": tp2["pred_bin"].to_numpy(),
}
overall_acc = {name: accuracy(y_true, pred) for name, pred in preds.items()}

seq_col = "Sequence_x" if "Sequence_x" in cask.columns else "Sequence"
seq_len = cask[seq_col].str.len().to_numpy()

bin_edges = np.percentile(seq_len, [0, 20, 40, 60, 80, 100])
bin_edges = np.unique(bin_edges)
bin_assign = pd.cut(seq_len, bins=bin_edges, include_lowest=True)
bin_cats = sorted(bin_assign.unique())

tick_labels = []
for iv in bin_cats:
    lo = int(np.floor(iv.left))
    hi = int(np.ceil(iv.right))
    tick_labels.append(f"{lo}–{hi}")

acc_by_len = {name: [] for name in preds}
bin_counts = []
for iv in bin_cats:
    mask = (bin_assign == iv)
    bin_counts.append(int(mask.sum()))
    for name, pred in preds.items():
        acc_by_len[name].append(accuracy(y_true[mask], pred[mask]))

summary_files = {
    "CASKnet": DATA_DIR / "CASKnet_weighted_bylength_summary.txt",
    "CNN": DATA_DIR / "cnn_bylength_summary.txt",
    "ProtT5": DATA_DIR / "prott5_linear_bylength_summary.txt",
    "Transformer": DATA_DIR / "transformer_bylength_summary.txt",
}
summary_data = {}
for name, file_path in summary_files.items():
    df = pd.read_csv(file_path, sep="\t")
    df = df[df["Range"] != "Overall"].copy()
    summary_data[name] = df

length_bins = summary_data["CASKnet"]["Range"].tolist()
x_mid = np.array([midpoint(x) for x in length_bins], dtype=float)

rows = []
for model, df in summary_data.items():
    for _, row in df.iterrows():
        roc, roc_sd = parse_mean_sd(row["ROC-AUC"])
        mcc, mcc_sd = parse_mean_sd(row["MCC"])
        rows.append({
            "Model": model,
            "Range": row["Range"],
            "ROC_AUC_mean": roc,
            "ROC_AUC_sd": roc_sd,
            "MCC_mean": mcc,
            "MCC_sd": mcc_sd,
        })
pd.DataFrame(rows).to_csv(OUT_DIR / "Fig2_panel_CD_AUC_MCC_values_used.csv", index=False)

pd.DataFrame({
    "model": list(overall_acc.keys()),
    "accuracy": [overall_acc[k] for k in overall_acc]
}).to_csv(OUT_DIR / "Fig2_panel_B_values_used.csv", index=False)

fig = plt.figure(figsize=(13.6, 9.0))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.04], hspace=0.43, wspace=0.21)

ax_a = fig.add_subplot(gs[0, 0])
rank_order = ["CASKnet", "SignalP6", "TargetP2"]
ypos = np.arange(len(rank_order))[::-1]
for y, name in zip(ypos, rank_order):
    val = overall_acc[name] * 100
    ax_a.barh(y, val, color=COLORS[name], height=0.42, edgecolor="white", linewidth=0.9, zorder=3)
    ax_a.scatter(val, y, s=32, color=COLORS[name], zorder=4)
    ax_a.text(val + 2.6, y, f"{val:.1f}%", va="center", ha="left", fontsize=10.5)
ax_a.set_yticks(ypos)
ax_a.set_yticklabels(rank_order)
ax_a.set_xlim(0, 105)
ax_a.set_xlabel("Accuracy (%)")
ax_a.set_title("Overall accuracy on secreted protein prediction", fontweight="bold", pad=12)
ax_a.grid(axis="x", zorder=0)
ax_a.spines["left"].set_position(("outward", 2))
ax_a.tick_params(axis="y", length=0)
ax_a.text(-0.13, 1.08, "a", transform=ax_a.transAxes, fontsize=22, fontweight="bold", va="bottom")

ax_b = fig.add_subplot(gs[0, 1])
xpos = np.arange(len(tick_labels))
plot_order = ["CASKnet", "SignalP6", "TargetP2"]
series_vals = {}
for name in plot_order:
    vals = np.array(acc_by_len[name])
    series_vals[name] = vals
    lw = 2.2 if name == "CASKnet" else 1.7
    ms = 5.2 if name == "CASKnet" else 4.8
    ax_b.plot(xpos, vals, color=COLORS[name], marker="o", markersize=ms, lw=lw, label=name, zorder=3)

for i in range(len(xpos)):
    point_vals = {name: series_vals[name][i] for name in plot_order}
    ordered = sorted(point_vals.items(), key=lambda kv: kv[1])
    spread = ordered[-1][1] - ordered[0][1]
    if spread < 0.08:
        placements = {
            ordered[0][0]: (-10, -13),
            ordered[1][0]: (0, -2),
            ordered[2][0]: (10, 10),
        }
    else:
        placements = {
            ordered[0][0]: (0, -13),
            ordered[1][0]: (0, 8),
            ordered[2][0]: (0, 10),
        }
    for name in plot_order:
        dx, dy = placements[name]
        ax_b.annotate(
            f"{point_vals[name]:.3f}",
            xy=(i, point_vals[name]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=8.1,
            color=COLORS[name],
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.82),
            zorder=5,
        )

for i, n in enumerate(bin_counts):
    ax_b.text(i, -0.14, f"n={n}", transform=ax_b.get_xaxis_transform(), ha="center", va="top", fontsize=8, color=SUBTLE)
ax_b.set_ylim(0, 1.04)
ax_b.set_xticks(xpos)
ax_b.set_xticklabels(tick_labels)
ax_b.set_xlabel("Sequence length (aa)", labelpad=14)
ax_b.set_ylabel("Accuracy")
ax_b.set_title("Accuracy across sequence-length bins", fontweight="bold", pad=12)
ax_b.grid(axis="both", zorder=0)
ax_b.legend(frameon=False, loc="center right")
ax_b.text(-0.12, 1.08, "b", transform=ax_b.transAxes, fontsize=22, fontweight="bold", va="bottom")


def plot_metric_panel(ax, metric, ylabel, letter):
    order = ["CASKnet", "CNN", "ProtT5", "Transformer"]
    for name in order:
        df = summary_data[name]
        means, sds = [], []
        for item in df[metric].tolist():
            m, sd = parse_mean_sd(item)
            means.append(m)
            sds.append(sd)
        means = np.array(means)
        sds = np.array(sds)

        if name == "CASKnet":
            lw, z, alpha, ms = 2.25, 6, 0.12, 3.4
        else:
            lw, z, alpha, ms = 1.65, 4, 0.09, 3.2

        ax.plot(
            x_mid, means,
            color=COLORS[name],
            linestyle=LINESTYLES[name],
            marker=MARKERS[name],
            markersize=ms,
            linewidth=lw,
            label=name,
            zorder=z,
        )
        ax.fill_between(
            x_mid,
            np.maximum(means - sds, -0.1),
            np.minimum(means + sds, 1.1),
            color=COLORS[name],
            alpha=alpha,
            linewidth=0,
            zorder=z - 1,
        )

    ax.set_xlim(x_mid.min() - 2, x_mid.max() + 2)
    ax.set_ylim(0.55 if metric == "ROC-AUC" else -0.10, 1.04)
    ax.set_xticks([8, 25, 45, 65, 85, 105])
    ax.set_xticklabels(["6–10", "21–30", "41–50", "61–70", "81–90", "101–110"])
    ax.set_xlabel("Sequence length (aa)")
    ax.set_ylabel(ylabel)
    ax.set_title(metric, fontweight="bold", pad=12)
    ax.grid(axis="both", zorder=0)
    ax.legend(frameon=False, loc="lower right")
    ax.text(-0.13, 1.08, letter, transform=ax.transAxes, fontsize=22, fontweight="bold", va="bottom")

ax_c = fig.add_subplot(gs[1, 0])
plot_metric_panel(ax_c, "ROC-AUC", "ROC-AUC", "c")

ax_d = fig.add_subplot(gs[1, 1])
plot_metric_panel(ax_d, "MCC", "MCC", "d")

fig.subplots_adjust(left=0.075, right=0.985, top=0.95, bottom=0.10)
fig.savefig(OUT_DIR / "Fig2_AUC_MCC.png", dpi=320, bbox_inches="tight", facecolor="white")
fig.savefig(OUT_DIR / "Fig2_AUC_MCC.svg", bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"Saved outputs to: {OUT_DIR}")
