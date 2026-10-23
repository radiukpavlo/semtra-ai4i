"""Figure style for the paper's data figures (Figures 2-4), matching the conference manuscript's house style.

Libertinus Sans 7.5 pt (DejaVu Sans when Libertinus is not installed), text written as outlines in SVG, Okabe-Ito
palette, full text width 15.8 cm, top and right spines off, panel letters "(a)", "(b)", ... and no titles.
`save_figure` writes PDF, SVG and a PNG fallback (280 dpi at 15.8 cm, about 1740 px wide) without changing the canvas
size, so text prints at 7.5 pt when the figure is placed 15.8 cm wide. Only the appearance is set here; what is
plotted is decided in make_report.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

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

TEXT_W_IN = 15.8 / 2.54      # CEUR-ART one-column text width
PNG_DPI = 280                # PNG fallback, about 1740 px across the text width
VECTOR_DPI = 600             # resolution of rasterized layers (dense scatter) inside PDF and SVG
FONT_DIRS = (Path("C:/Windows/Fonts"), Path.home() / "AppData/Local/Microsoft/Windows/Fonts",
             Path("/usr/share/fonts"), Path.home() / ".fonts")


def _font_family() -> str:
    for d in FONT_DIRS:
        if d.exists():
            for f in d.rglob("LibertinusSans-*.ttf"):
                font_manager.fontManager.addfont(str(f))
    names = {f.name for f in font_manager.fontManager.ttflist}
    return "Libertinus Sans" if "Libertinus Sans" in names else "DejaVu Sans"


def apply_paper_style() -> str:
    fam = _font_family()
    mpl.rcParams.update({
        "font.family": fam,
        "mathtext.fontset": "custom",
        "mathtext.rm": fam,
        "mathtext.it": f"{fam}:italic",
        "mathtext.bf": f"{fam}:bold",
        "svg.fonttype": "path",          # text as outlines: the SVG needs no fonts to render
        "svg.hashsalt": "ai4i-figures",  # stable ids between runs
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 7.5,
        "axes.titlelocation": "left",
        "axes.titlepad": 3.0,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.unicode_minus": True,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "legend.fontsize": 7.0,
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 1.1,
        "legend.borderaxespad": 0.3,
        "hatch.linewidth": 0.5,
        "figure.dpi": 150,
        "figure.constrained_layout.h_pad": 0.03,
        "figure.constrained_layout.w_pad": 0.03,
        "savefig.dpi": PNG_DPI,
        "savefig.bbox": None,            # keep the canvas exactly TEXT_W_IN wide
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    return fam


apply_nature_style = apply_paper_style   # earlier name, kept for old callers


def paper_figure(height_in: float, ncols: int = 1, **kw):
    """Full-text-width figure with constrained layout (labels and legends stay inside the canvas)."""
    return plt.subplots(1, ncols, figsize=(TEXT_W_IN, height_in), layout="constrained", **kw)


def size_mm(width_mm: float, height_mm: float) -> tuple[float, float]:
    return width_mm / 25.4, height_mm / 25.4


def style_axis(ax, grid_axis: str | None = None) -> None:
    ax.tick_params(direction="out", length=2.5, width=0.7, pad=2)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=PALETTE["light"], linewidth=0.4, alpha=0.8)
        ax.set_axisbelow(True)


def add_panel_label(ax, label: str) -> None:
    """Panel letter "(a)" at the top left of the axes, in the title slot (no titles are used)."""
    ax.set_title(label if label.startswith("(") else f"({label})", loc="left", fontsize=7.5, color=PALETTE["black"])


def thousands_formatter() -> FuncFormatter:
    """Tick labels in the paper's number style: 9000, 10,000 (separator from five digits on)."""
    return FuncFormatter(lambda v, _pos: f"{v:,.0f}" if abs(v) >= 10000 else f"{v:.0f}")


def save_figure(fig, path: Path | str, formats: Sequence[str] = ("pdf", "svg", "png"), dpi: int | None = None,
                layout: bool = True) -> list[str]:
    stem = Path(path).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    if layout and fig.get_layout_engine() is None:
        fig.tight_layout(pad=0.7)
    meta = {"svg": {"Date": None}, "pdf": {"CreationDate": None}, "png": {}}
    saved = []
    for fmt in formats:
        out = stem.with_suffix(f".{fmt}")
        res = dpi or (PNG_DPI if fmt == "png" else VECTOR_DPI)
        fig.savefig(out, facecolor="white", dpi=res, metadata=meta.get(fmt, {}))
        saved.append(str(out))
    plt.close(fig)
    return saved
