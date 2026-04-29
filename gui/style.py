"""Research-paper style for matplotlib figures.

Sets a single set of rcParams (Okabe-Ito categorical palette, cividis
sequential cmap, serif fonts, light grid, clean spines) so every
``plt.subplots`` call across the GUI inherits a consistent look. Call
``apply()`` once at app start.
"""

import matplotlib as mpl


# Okabe-Ito 8-colour palette: colour-blind safe, recommended by Nature
# and widely used in IEEE / journal figures.
PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#F0E442",  # yellow
    "#000000",  # black
]


def apply():
    mpl.rcParams.update({
        # Typography
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,

        # Axes & spines
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#222222",
        "axes.labelcolor": "#222222",

        # Grid
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "grid.color": "#888888",

        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.color": "#222222",
        "ytick.color": "#222222",

        # Lines & markers
        "lines.linewidth": 1.5,
        "lines.markersize": 4,

        # Legend
        "legend.frameon": False,
        "legend.borderaxespad": 0.4,

        # Figure & saved output
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "figure.facecolor": "white",
        "axes.facecolor": "white",

        # Colour cycle (lines) and default sequential colormap (heatmaps)
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "image.cmap": "cividis",
    })
