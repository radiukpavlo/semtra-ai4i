"""House figure style of the SEMTRA project (adapted from scripts/core/figure_style.py).

Same rcParams and Okabe-Ito palette; `save_figure` writes PDF and SVG to the given path and does not
rewrite titles or font sizes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt

PALETTE = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "teal": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#1F1F1F",
    "dark": "#4D4D4D",
    "mid": "#8F8F8F",
    "light": "#D8D8D8",
    "pale_blue": "#DDECF7",
    "pale_teal": "#DDEFE8",
    "pale_orange": "#F8EAC7",
}


def apply_nature_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
        "mathtext.fontset": "dejavusans",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7.6,
        "axes.labelsize": 7.6,
        "axes.titlesize": 7.6,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.fontsize": 6.8,
        "legend.frameon": False,
        "figure.dpi": 180,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def size_mm(width_mm: float, height_mm: float) -> tuple[float, float]:
    return width_mm / 25.4, height_mm / 25.4


def style_axis(ax, grid_axis: str | None = None) -> None:
    ax.tick_params(direction="out", length=2.5, width=0.55, pad=2)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=PALETTE["light"], linewidth=0.35, alpha=0.65)
        ax.set_axisbelow(True)


def add_panel_label(ax, label: str, x: float = -0.12, y: float = 1.02) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, fontweight="bold",
            color=PALETTE["black"])


def save_figure(fig, path: Path | str, formats: Sequence[str] = ("pdf", "svg"), dpi: int = 600,
                layout: bool = True) -> list[str]:
    stem = Path(path).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    if layout:
        fig.tight_layout(pad=0.7)
    saved = []
    for fmt in formats:
        out = stem.with_suffix(f".{fmt}")
        fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=dpi)
        saved.append(str(out))
    plt.close(fig)
    return saved
