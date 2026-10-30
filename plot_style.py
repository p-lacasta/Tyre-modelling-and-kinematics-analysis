#---------------------------------------------------------
# OXFORD BROOKES UNIVERSITY - Motorsport Engineering MSc
# Advanced Vehicle dynamics
#
# Coursework 1 - "Tyre modelling and kinematic analysis"
#---------------------------------------------------------

#---------------------------------------------------------
# SHARED PLOT STYLE (used by every part of the project)
#---------------------------------------------------------

import os
import matplotlib as mpl
import matplotlib.pyplot as plt

#---------------------------------------------------------
# Colour palette
#---------------------------------------------------------
# Categorical hues in fixed order (colour-blind safe as a set of three).
# Text and reference lines use neutral inks, never a series colour.

PRIMARY = "#003f5c"    # deep navy: measured data / exact solution / displaced geometry
SECONDARY = "#ef5675"  # coral: fitted model / first-order estimate / migration
TERTIARY = "#7a5195"   # plum: third series (e.g. swing arm length)

INK = "#1e293b"        # primary text
INK_2 = "#475569"      # secondary text, axes
SLATE = "#8796a8"      # static / reference geometry
MUTED = "#a9b4c2"      # reference lines (centreline, zero lines)
GRID = "#e2e8f0"
SURFACE = "#ffffff"
GROUND = "#334155"     # ground plane

DPI = 300
FIG_SINGLE = (6.4, 4.4)   # one plot per figure [in]


def apply_style():
    """Set the project-wide matplotlib defaults."""
    mpl.rcParams.update({
        # Fonts
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "figure.titleweight": "bold",

        # Colours
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "axes.prop_cycle": mpl.cycler(color=[PRIMARY, SECONDARY, TERTIARY]),

        # Axes and grid
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,

        # Lines and markers
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
        "scatter.edgecolors": PRIMARY,

        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": GRID,
        "legend.fancybox": False,
        "legend.borderpad": 0.5,

        # Output
        "figure.dpi": 110,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def figures_dir(script_file):
    """`figures/` folder next to the calling script (created if missing)."""
    path = os.path.join(os.path.dirname(os.path.abspath(script_file)), "figures")
    os.makedirs(path, exist_ok=True)
    return path


def save_figure(fig, script_file, name):
    fig.savefig(os.path.join(figures_dir(script_file), name))


def info_box(ax, text, loc="upper left"):
    """Neutral annotation box in an axes corner (e.g. goodness-of-fit metrics)."""
    x = 0.03 if "left" in loc else 0.97
    y = 0.96 if "upper" in loc else 0.04
    ax.text(x, y, text, transform=ax.transAxes, fontsize=9, color=INK,
            ha="left" if "left" in loc else "right",
            va="top" if "upper" in loc else "bottom",
            bbox=dict(boxstyle="square,pad=0.45", facecolor=SURFACE, edgecolor=GRID, alpha=0.92))


def plot_curve_fit(x, y, x_fit, y_fit, rmse, r_squared, title, xlabel, ylabel,
                   rmse_unit="", metrics_loc="upper left", legend_loc="lower right"):
    """Standard 'measured points + fitted curve' figure shared by Parts 0 and 1."""
    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    ax.scatter(x, y, s=45, facecolor=SURFACE, edgecolors=PRIMARY, linewidths=1.6,
               label="Measured data", zorder=3)
    ax.plot(x_fit, y_fit, color=SECONDARY, lw=2, label="Fitted model", zorder=2)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc=legend_loc)

    unit = f" {rmse_unit}" if rmse_unit else ""
    info_box(ax, f"RMSE = {rmse:.3g}{unit}\n$R^2$ = {r_squared:.4f}", loc=metrics_loc)

    fig.tight_layout()
    return fig, ax
