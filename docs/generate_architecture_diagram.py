"""
Generate the Lich Theory Architecture Diagram.

Architecture:
  Input -> Lich (Router) -> Headless Models (no attention layers) -> Synthesizer -> Phylactery (Agent) -> Output
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

fig, ax = plt.subplots(figsize=(16, 9), facecolor='#0d1117')
ax.set_facecolor('#0d1117')
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')

# --- Color palette ---
COL_INPUT    = '#58a6ff'
COL_LICH     = '#f85149'
COL_HEADLESS = '#8b949e'
COL_SYNTH    = '#d29922'
COL_PHYLAC   = '#a371f7'
COL_OUTPUT   = '#3fb950'
COL_BORDER   = '#30363d'
COL_TEXT     = '#e6edf3'
COL_SUBTEXT  = '#8b949e'
COL_ARROW    = '#484f58'
COL_GLOW     = '#1f6feb'

def draw_box(ax, x, y, w, h, color, label, sublabel=None, rounded=True):
    """Draw a styled box with label."""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.15" if rounded else "square,pad=0.05",
        facecolor=color + '18',
        edgecolor=color,
        linewidth=2.5,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y + (0.12 if sublabel else 0), label,
            fontsize=13, fontweight='bold', color=COL_TEXT,
            ha='center', va='center', zorder=4,
            fontfamily='monospace')
    if sublabel:
        ax.text(x, y - 0.32, sublabel,
                fontsize=8.5, color=color, alpha=0.85,
                ha='center', va='center', zorder=4,
                fontfamily='monospace')

def draw_arrow(ax, x1, y1, x2, y2, color=COL_ARROW, lw=2, style='-|>'):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle=style,
                    color=color,
                    lw=lw,
                    connectionstyle='arc3,rad=0',
                ),
                zorder=2)

# ===== Title =====
ax.text(8, 8.35, 'LICH THEORY  //  PROPOSED ARCHITECTURE',
        fontsize=18, fontweight='bold', color=COL_TEXT,
        ha='center', va='center', fontfamily='monospace', zorder=5)
ax.text(8, 7.85, 'Attention-routed headless model orchestration with persistent identity',
        fontsize=9.5, color=COL_SUBTEXT,
        ha='center', va='center', fontfamily='monospace', zorder=5)

# Thin divider line
ax.plot([1.5, 14.5], [7.5, 7.5], color=COL_BORDER, linewidth=1, zorder=1)

# ===== Layout coordinates =====
Y_MAIN = 4.5
Y_MODELS_TOP = 5.6
Y_MODELS_MID = 4.5
Y_MODELS_BOT = 3.4

X_INPUT  = 1.8
X_LICH   = 4.5
X_M1     = 7.5
X_M2     = 7.5
X_M3     = 7.5
X_SYNTH  = 10.5
X_PHYLAC = 13.0
X_OUTPUT = 15.2

# ===== Draw boxes =====

# INPUT
draw_box(ax, X_INPUT, Y_MAIN, 2.0, 1.2, COL_INPUT, 'INPUT', 'User Query')

# LICH (Router)
draw_box(ax, X_LICH, Y_MAIN, 2.2, 1.6, COL_LICH, 'LICH', 'Attention Router')

# Headless Models (3 stacked)
draw_box(ax, X_M1, Y_MODELS_TOP, 2.4, 0.85, COL_HEADLESS, 'MODEL  A', 'Headless / No Attn')
draw_box(ax, X_M2, Y_MODELS_MID, 2.4, 0.85, COL_HEADLESS, 'MODEL  B', 'Headless / No Attn')
draw_box(ax, X_M3, Y_MODELS_BOT, 2.4, 0.85, COL_HEADLESS, 'MODEL  C', 'Headless / No Attn')

# Dots to suggest more models
ax.text(X_M3, Y_MODELS_BOT - 0.72, '. . .', fontsize=12, color=COL_SUBTEXT,
        ha='center', va='center', fontfamily='monospace', zorder=4)

# SYNTHESIZER
draw_box(ax, X_SYNTH, Y_MAIN, 2.2, 1.6, COL_SYNTH, 'SYNTHESIZER', 'Blend & Merge')

# PHYLACTERY (Agent)
draw_box(ax, X_PHYLAC, Y_MAIN, 2.2, 1.6, COL_PHYLAC, 'PHYLACTERY', 'Persistent Agent')

# OUTPUT
draw_box(ax, X_OUTPUT, Y_MAIN, 1.4, 1.2, COL_OUTPUT, 'OUT', 'Response')

# ===== Draw arrows =====

# Input -> Lich
draw_arrow(ax, X_INPUT + 1.05, Y_MAIN, X_LICH - 1.15, Y_MAIN, COL_INPUT)

# Lich -> Headless Models (fan out)
draw_arrow(ax, X_LICH + 1.15, Y_MAIN + 0.35, X_M1 - 1.25, Y_MODELS_TOP, COL_LICH)
draw_arrow(ax, X_LICH + 1.15, Y_MAIN,        X_M2 - 1.25, Y_MODELS_MID, COL_LICH)
draw_arrow(ax, X_LICH + 1.15, Y_MAIN - 0.35, X_M3 - 1.25, Y_MODELS_BOT, COL_LICH)

# Headless Models -> Synthesizer (fan in)
draw_arrow(ax, X_M1 + 1.25, Y_MODELS_TOP, X_SYNTH - 1.15, Y_MAIN + 0.35, COL_SYNTH)
draw_arrow(ax, X_M2 + 1.25, Y_MODELS_MID, X_SYNTH - 1.15, Y_MAIN,        COL_SYNTH)
draw_arrow(ax, X_M3 + 1.25, Y_MODELS_BOT, X_SYNTH - 1.15, Y_MAIN - 0.35, COL_SYNTH)

# Synthesizer -> Phylactery
draw_arrow(ax, X_SYNTH + 1.15, Y_MAIN, X_PHYLAC - 1.15, Y_MAIN, COL_SYNTH)

# Phylactery -> Output
draw_arrow(ax, X_PHYLAC + 1.15, Y_MAIN, X_OUTPUT - 0.75, Y_MAIN, COL_PHYLAC)

# Phylactery feedback loop (identity persistence) — curved arrow back to Lich
ax.annotate('', xy=(X_LICH, Y_MAIN - 0.85), xytext=(X_PHYLAC, Y_MAIN - 0.85),
            arrowprops=dict(
                arrowstyle='-|>',
                color=COL_PHYLAC,
                lw=1.5,
                linestyle='dashed',
                connectionstyle='arc3,rad=0.35',
            ), zorder=2)
ax.text((X_LICH + X_PHYLAC) / 2, 2.15, 'Identity Feedback Loop',
        fontsize=8, color=COL_PHYLAC, alpha=0.7,
        ha='center', va='center', fontfamily='monospace',
        style='italic', zorder=4)

# ===== Legend at bottom =====
legend_y = 1.0
legend_items = [
    (COL_LICH,     'Lich',        'Central attention basin acts as intelligent router'),
    (COL_HEADLESS, 'Headless',    'Specialist models with attention layers removed'),
    (COL_SYNTH,    'Synthesizer', 'Merges multi-model outputs into coherent response'),
    (COL_PHYLAC,   'Phylactery',  'Persistent agent identity — survives sessions'),
]

start_x = 1.8
for i, (col, title, desc) in enumerate(legend_items):
    bx = start_x + i * 3.5
    ax.plot(bx, legend_y, 's', color=col, markersize=8, zorder=4)
    ax.text(bx + 0.25, legend_y, f'{title}:', fontsize=8.5, fontweight='bold',
            color=COL_TEXT, va='center', fontfamily='monospace', zorder=4)
    ax.text(bx + 0.25, legend_y - 0.35, desc, fontsize=7, color=COL_SUBTEXT,
            va='center', fontfamily='monospace', zorder=4)

# Save
output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, 'architecture.png')
fig.savefig(output_path, facecolor='#0d1117', edgecolor='none',
            bbox_inches='tight', dpi=180, pad_inches=0.3)
plt.close('all')
print(f"Saved: {output_path}")
