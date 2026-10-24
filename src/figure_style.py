"""Figure style for the paper's data figures (Figures 2-4), matching the conference manuscript's house style.

Libertinus Sans 7.5 pt (DejaVu Sans when Libertinus is not installed), text written as outlines in SVG, Okabe-Ito
palette, full text width 15.8 cm, top and right spines off, panel letters "(a)", "(b)", ... and no titles.
`save_figure` writes PDF, SVG and a PNG fallback (280 dpi at 15.8 cm, about 1740 px wide) without changing the canvas
size, so text prints at 7.5 pt when the figure is placed 15.8 cm wide. Panels can also be saved as files of their own
(`panel_figure`, `fit_panel_row`, `legend_strip`), without letters, for a table of sub-figures in which every file is
placed at its native size, so the text prints at the same size as in the combined figure. Only the appearance is set
here; what is plotted is decided in make_report.py.
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

TEXT_W_CM = 15.8             # CEUR-ART one-column text width
TEXT_W_IN = TEXT_W_CM / 2.54
PAD_IN = 0.03                # outer pad of a separate panel file, equal to the constrained-layout pad
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


def axes_size_cm(fig) -> list[tuple[float, float]]:
    """Width and height (cm) of every axes box of a figure after its layout has run."""
    fig.draw_without_rendering()
    w, h = fig.get_size_inches()
    return [(ax.get_position().width * w * 2.54, ax.get_position().height * h * 2.54) for ax in fig.axes]


def panel_figure():
    """Figure and axes for one panel saved as a file of its own; `fit_panel_row` sets its final size."""
    fig = plt.figure(figsize=(TEXT_W_IN / 2, 2.5))
    return fig, fig.add_axes((0.2, 0.2, 0.6, 0.6))


def _margins(fig, ax) -> tuple[float, float, float, float]:
    """Room (inches) taken by tick labels, axis labels and legend left of, right of, below and above the axes box."""
    fig.draw_without_rendering()
    w, h = fig.get_size_inches()
    box = ax.get_position()
    tight = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    return box.x0 * w - tight.x0, tight.x1 - box.x1 * w, box.y0 * h - tight.y0, tight.y1 - box.y1 * h


def fit_panel_row(panels, width_cm: float, axes_w_cm, axes_h_cm: float) -> None:
    """Size the separate panel files of one figure row, to be set side by side at their native size.

    Every canvas is `width_cm` wide and all share one height; every axes box is `axes_h_cm` tall at the same distance
    from the bottom edge, so the x axes line up. An axes box is `axes_w_cm` wide (one value, or one per panel) or, where
    its tick labels, axis labels and legend would not fit, as wide as they leave room for; the content is centred.
    Font sizes and everything plotted stay as drawn. Save the panels with `save_figure(..., layout=False)`.
    """
    widths = list(axes_w_cm) if isinstance(axes_w_cm, (list, tuple)) else [axes_w_cm] * len(panels)
    widths = [v / 2.54 for v in widths]
    W, ah = width_cm / 2.54, axes_h_cm / 2.54
    for (fig, ax), aw in zip(panels, widths):   # first guess at the target axes size, so the tick locators see it
        fig.set_size_inches(W, ah + 1.0)
        ax.set_position((0.2, 0.5 / (ah + 1.0), min(aw, 0.7 * W) / W, ah / (ah + 1.0)))
    for _ in range(4):
        m = [_margins(fig, ax) for fig, ax in panels]
        bottom = PAD_IN + max(r[2] for r in m)
        H = bottom + ah + max(r[3] for r in m) + PAD_IN
        for (fig, ax), (left, right, _, _), aw in zip(panels, m, widths):
            room = W - 2 * PAD_IN - left - right
            w = min(aw, room)
            fig.set_size_inches(W, H)
            ax.set_position(((PAD_IN + left + (room - w) / 2) / W, bottom / H, w / W, ah / H))
    for fig, ax in panels:                      # everything drawn must lie on the canvas
        left, right, below, above = _margins(fig, ax)
        box = ax.get_position()
        if min(box.x0 * W - left, W - box.x1 * W - right, box.y0 * H - below, H - box.y1 * H - above) < -1e-3:
            raise RuntimeError("panel content does not fit its canvas")


def legend_strip(handles, width_cm: float, ncol: int):
    """Canvas `width_cm` wide holding only a one-row figure legend, centred, with the outer pad above and below."""
    fig = plt.figure(figsize=(width_cm / 2.54, 1.0))
    legend = fig.legend(handles=handles, loc="center", ncol=ncol)
    fig.draw_without_rendering()
    height = legend.get_window_extent(fig.canvas.get_renderer()).height / fig.dpi
    fig.set_size_inches(width_cm / 2.54, height + 2 * PAD_IN)
    return fig


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
