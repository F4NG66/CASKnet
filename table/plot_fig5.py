"""
Fig 5 — Model performance comparison
  fig5_bylength.png  – line plots (mean ± std shading) for each metric across
                       sequence-length bins, all 4 models overlaid
  fig5_overall.png   – formatted table of Overall (5-fold CV) metrics
"""

import os, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── model definitions ───────────────────────────────────────────────────────
MODELS = {
    'CASKnet'    : 'CASKnet_weighted_bylength_summary.txt',
    'CNN'        : 'cnn_bylength_summary.txt',
    'ProtT5'     : 'prott5_linear_bylength_summary.txt',
    'Transformer': 'transformer_bylength_summary.txt',
}

COLORS = {
    'CASKnet'    : '#1565C0',   # deep blue  (highlight – our model)
    'CNN'        : '#E64A19',   # deep orange
    'ProtT5'     : '#2E7D32',   # deep green
    'Transformer': '#7B1FA2',   # deep purple
}

LINESTYLES = {
    'CASKnet'    : '-',
    'CNN'        : '--',
    'ProtT5'     : '-.',
    'Transformer': ':',
}

LINEWIDTHS = {
    'CASKnet'    : 2.4,
    'CNN'        : 1.8,
    'ProtT5'     : 1.8,
    'Transformer': 1.8,
}

METRICS = ['Accuracy', 'Sensitivity', 'Specificity', 'ROC-AUC',
           'PR-AUC', 'F1-score', 'MCC']

LENGTH_BINS = [
    '6-10','11-15','16-20','21-25','26-30','31-35','36-40','41-45',
    '46-50','51-55','56-60','61-65','66-70','71-75','76-80','81-85',
    '86-90','91-95','96-100','101-105','106-110',
]
# use midpoints on x-axis
BIN_MID = [int(b.split('-')[0]) + 2 for b in LENGTH_BINS]


# ── parser ───────────────────────────────────────────────────────────────────
def parse_file(path):
    """Return dict  range_label -> {metric: (mean, std)}."""
    data = {}
    with open(path) as fh:
        lines = [l.strip() for l in fh if l.strip()]
    header = lines[0].split('\t')
    for line in lines[1:]:
        parts = line.split('\t')
        label = parts[0]
        row = {}
        for idx, col in enumerate(header[1:], 1):
            if idx >= len(parts):
                row[col] = (0.0, 0.0)
                continue
            m = re.match(r'([-\d.]+)±([\d.]+)', parts[idx])
            if m:
                row[col] = (float(m.group(1)), float(m.group(2)))
            else:
                try:
                    row[col] = (float(parts[idx]), 0.0)
                except ValueError:
                    row[col] = (0.0, 0.0)
        data[label] = row
    return data


all_data = {name: parse_file(os.path.join(BASE_DIR, fname))
            for name, fname in MODELS.items()}

plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.linewidth'   : 1.1,
    'figure.dpi'       : 150,
    'grid.alpha'       : 0.35,
    'grid.linewidth'   : 0.7,
})


# ════════════════════════════════════════════════════════════════════════════
# Figure 1 – by-length line plots
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 11))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.35)

axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(3)]

for ax_idx, metric in enumerate(METRICS):
    ax = axes[ax_idx]
    for model_name in MODELS:
        means, stds = [], []
        for b in LENGTH_BINS:
            mean, std = all_data[model_name][b].get(metric, (0, 0))
            means.append(mean)
            stds.append(std)
        means, stds = np.array(means), np.array(stds)
        ax.plot(BIN_MID, means,
                color=COLORS[model_name],
                linestyle=LINESTYLES[model_name],
                linewidth=LINEWIDTHS[model_name],
                label=model_name)
        ax.fill_between(BIN_MID, means - stds, means + stds,
                        color=COLORS[model_name], alpha=0.10)

    ax.set_title(metric, fontsize=12, fontweight='bold', pad=4)
    ax.set_xlabel('Sequence length (aa)', fontsize=9)
    ax.set_ylabel(metric, fontsize=9)
    ax.set_xlim(BIN_MID[0] - 2, BIN_MID[-1] + 2)
    ax.set_xticks(BIN_MID[::2])
    ax.set_xticklabels([str(x) for x in BIN_MID[::2]], fontsize=7.5, rotation=35)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.grid(axis='y')

    # highlight CASKnet line on top
    # (already drawn last if we care about z-order; reorder by redrawing)

# 8th panel → shared legend
legend_ax = axes[7]
legend_ax.axis('off')
handles = [
    Line2D([0], [0],
           color=COLORS[m], linewidth=LINEWIDTHS[m],
           linestyle=LINESTYLES[m], label=m)
    for m in MODELS
]
legend_ax.legend(handles=handles, loc='center', fontsize=12,
                 frameon=True, framealpha=0.9,
                 title='Model', title_fontsize=12)

# 9th panel → small note
note_ax = axes[8]
note_ax.axis('off')
note_ax.text(0.5, 0.5,
             'Shaded region = ±1 SD\n(5-fold cross-validation)',
             ha='center', va='center', fontsize=10, color='grey',
             transform=note_ax.transAxes)

fig.suptitle('Model performance across sequence length bins',
             fontsize=15, fontweight='bold', y=1.01)

out_len = os.path.join(BASE_DIR, 'fig5_bylength.png')
plt.savefig(out_len, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved  {out_len}')


# ════════════════════════════════════════════════════════════════════════════
# Figure 2 – Overall metrics table
# ════════════════════════════════════════════════════════════════════════════
model_names = list(MODELS.keys())

# build cell text and numeric matrix for best-cell highlighting
cell_text   = []
cell_means  = []      # for finding column maxima
for model in model_names:
    row_text, row_num = [], []
    for metric in METRICS:
        mean, std = all_data[model]['Overall'].get(metric, (0, 0))
        row_text.append(f'{mean:.4f}\n±{std:.4f}')
        row_num.append(mean)
    cell_text.append(row_text)
    cell_means.append(row_num)

cell_means = np.array(cell_means)   # shape (4, 7)

# colour each cell: best model per metric = light tint, others = white
BG_BEST  = '#BBDEFB'   # light blue
BG_NORM  = '#FFFFFF'
BG_HEAD  = '#1565C0'
TXT_HEAD = '#FFFFFF'

cell_colors = []
for i in range(len(model_names)):
    row_c = []
    for j in range(len(METRICS)):
        row_c.append(BG_BEST if i == np.argmax(cell_means[:, j]) else BG_NORM)
    cell_colors.append(row_c)

fig, ax = plt.subplots(figsize=(16, 3.5))
ax.axis('off')

col_labels_display = ['Accuracy', 'Sensitivity', 'Specificity',
                      'ROC-AUC', 'PR-AUC', 'F1-score', 'MCC']
row_labels_display = model_names

tbl = ax.table(
    cellText=cell_text,
    rowLabels=row_labels_display,
    colLabels=col_labels_display,
    cellColours=cell_colors,
    loc='center',
    cellLoc='center',
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.0, 2.4)

# style header row
for j in range(len(METRICS)):
    cell = tbl[(0, j)]
    cell.set_facecolor(BG_HEAD)
    cell.get_text().set_color(TXT_HEAD)
    cell.get_text().set_fontweight('bold')
    cell.get_text().set_fontsize(10)

# style row-label column
for i, model in enumerate(model_names):
    cell = tbl[(i + 1, -1)]
    cell.set_facecolor(COLORS[model])
    cell.get_text().set_color('white')
    cell.get_text().set_fontweight('bold')
    cell.get_text().set_fontsize(10)

# bold the best value in each metric column
for j in range(len(METRICS)):
    best_i = int(np.argmax(cell_means[:, j]))
    tbl[(best_i + 1, j)].get_text().set_fontweight('bold')
    tbl[(best_i + 1, j)].get_text().set_fontsize(11)

# legend note
ax.text(0.5, -0.06,
        '★ Highlighted cell = best value per metric  |  Values: mean ± std (5-fold CV)',
        ha='center', va='top', transform=ax.transAxes,
        fontsize=9.5, color='grey', style='italic')

ax.set_title('Overall model performance  (5-fold cross-validation)',
             fontsize=13, fontweight='bold', pad=14)

out_ov = os.path.join(BASE_DIR, 'fig5_overall.png')
plt.savefig(out_ov, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved  {out_ov}')

print('\nDone.')
