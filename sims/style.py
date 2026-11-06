"""Figure style for the BB601 Team 10 simulations.

Adapted from the PH561 project's simulations/common.py (same fonts, colours,
layout check and report-width two-panel layout), so the figures look the same.
Every figure is written to sims/figs/<name>.png (300 dpi, for Google Docs and
slides) and sims/figs/<name>.pdf (vector).
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.container import BarContainer
from matplotlib.text import Text
from matplotlib.transforms import Bbox

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figs")

# Colours mean the same thing on every plot.
BLUE = "#1F4E79"         # our design / the model
CRIMSON = "#9E1B32"      # the problem / limits / literature condition
DARK_CRIMSON = "#6E1323"
BLUE_TINT = "#DCE8F4"
CRIMSON_TINT = "#F3DCE0"
GREY = "0.4"
BAND = "0.92"            # background band, e.g. the 800-830 nm trigger window
ORANGE = CRIMSON         # alias kept from PH561 naming

_STIX = "/System/Library/Fonts/Supplemental/STIXTwoText.ttf"
_STIX_MATH = "/System/Library/Fonts/Supplemental/STIXTwoMath.otf"


def setup(font=15):
    for f in (_STIX, _STIX_MATH):
        if os.path.exists(f):
            font_manager.fontManager.addfont(f)
    family = "STIX Two Text" if os.path.exists(_STIX) else "STIXGeneral"
    plt.rcParams.update({
        "font.size": font, "axes.labelsize": font, "axes.titlesize": font,
        "xtick.labelsize": 0.9 * font, "ytick.labelsize": 0.9 * font,
        "legend.fontsize": 0.85 * font, "legend.frameon": False,
        "legend.handlelength": 1.8, "legend.borderaxespad": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlepad": 9, "axes.labelpad": 5,
        # STIX Two Math as fallback for arrows and other symbols STIX Two Text lacks
        "font.family": [family, "STIX Two Math", "STIXGeneral"],
        "mathtext.fontset": "stix", "axes.titleweight": "normal",
        "pdf.fonttype": 42, "savefig.dpi": 300,
    })



# ---------------------------------------------------------------------------
# layout check
# ---------------------------------------------------------------------------
def _hidden_ticklabels(fig):
    """Tick labels that exist but are not drawn (outside the axis limits)."""
    hidden = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            drawn = set(axis._update_ticks())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if tick not in drawn:
                    hidden.update((tick.label1, tick.label2))
    return hidden


def _texts(fig):
    out = []
    hidden = _hidden_ticklabels(fig)
    for t in fig.findobj(Text):
        if t in hidden or (t.axes is not None and not t.axes.get_visible()):
            continue
        if t.get_visible() and t.get_text().strip() and t.get_alpha() != 0:
            try:
                bb = Text.get_window_extent(t)     # for annotations: the words, not the arrow
            except Exception:
                continue
            if bb.width > 0 and bb.height > 0:
                out.append((t, bb))
    return out


def _in_legend(artist, fig):
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is not None and artist in leg.get_texts() + [leg.get_title()]:
            return leg
    for leg in fig.legends:
        if artist in leg.get_texts() + [leg.get_title()]:
            return leg
    return None


def _line_points(line):
    """Sample a Line2D in display coordinates, about one point every 2 px."""
    xy = np.asarray(line.get_xydata(), float)
    if xy.size == 0:
        return np.empty((0, 2)), np.empty((0, 2))
    p = line.get_transform().transform(xy)
    p = p[np.all(np.isfinite(p), axis=1)]
    marks = p if line.get_marker() not in (None, "None", "", " ") else np.empty((0, 2))
    if line.get_linestyle() in ("None", "", " ") or len(p) < 2:
        return np.empty((0, 2)), marks
    seg = np.diff(p, axis=0)
    n = np.maximum(1, np.ceil(np.hypot(*seg.T) / 2).astype(int))
    idx = np.repeat(np.arange(len(seg)), n)
    frac = np.concatenate([np.arange(k) / k for k in n])
    pts = p[idx] + seg[idx] * frac[:, None]
    return np.vstack([pts, p[-1:]]), marks


def _hits(bb, pts, grow=0.0):
    if len(pts) == 0:
        return False
    x0, y0, x1, y1 = bb.x0 + 1 - grow, bb.y0 + 1 - grow, bb.x1 - 1 + grow, bb.y1 - 1 + grow
    return bool(np.any((pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)))


def _overlap(a, b, tol=1.0):
    return (min(a.x1, b.x1) - max(a.x0, b.x0) > tol) and (min(a.y1, b.y1) - max(a.y0, b.y0) > tol)


def check_layout(fig, name, quiet=False):
    """Report text-on-text, text-on-curve, legend-on-curve and panel-on-panel
    collisions. Returns the list of problems (empty when the layout is clean)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.canvas.draw()
    problems = ["missing glyph: %s" % w.message for w in caught if "missing from" in str(w.message)]
    texts = _texts(fig)
    # legends count as one box each
    boxes, seen = [], set()
    for t, bb in texts:
        leg = _in_legend(t, fig)
        if leg is not None:
            if id(leg) not in seen:
                seen.add(id(leg))
                boxes.append(("legend", leg.get_window_extent(), leg.axes, leg))
        else:
            boxes.append((t.get_text().replace("\n", " ")[:40], bb, t.axes, t))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _overlap(boxes[i][1].padded(3), boxes[j][1].padded(3)):
                problems.append("text '%s' overlaps text '%s'" % (boxes[i][0], boxes[j][0]))

    axes = [ax for ax in fig.axes if ax.get_visible()]
    axes += [c for ax in list(axes) for c in ax.child_axes if c.get_visible()]
    parent = {}
    for ax in axes:
        bb = ax.get_window_extent()
        for o in axes:
            ob = o.get_window_extent()
            if o is not ax and ob.x0 <= bb.x0 + 1 and ob.y0 <= bb.y0 + 1 and ob.x1 >= bb.x1 - 1 \
                    and ob.y1 >= bb.y1 - 1 and (ob.width > bb.width + 2 or ob.height > bb.height + 2):
                parent[ax] = o
    main = [ax for ax in axes if ax not in parent]
    for ax, par in parent.items():            # an inset must not sit on its parent's curves
        for line in par.get_lines():
            pts, marks = _line_points(line)
            if line.get_visible() and (_hits(ax.get_window_extent(), pts) or _hits(ax.get_window_extent(), marks)):
                problems.append("inset in '%s' covers a curve" % par.get_title()[:30])
                break

    def home(ax):
        return parent.get(ax, ax) if ax is not None else None

    for label, bb, owner, art in boxes:
        if not (owner is None or label == "legend" or art in owner.texts):
            continue                           # tick and axis labels sit outside the data
        for ax in ([owner] if owner is not None else axes):
            for line in ax.get_lines():
                if not line.get_visible() or line.get_label() == "_guide":
                    continue
                pts, marks = _line_points(line)
                if line.get_clip_on():
                    ab = ax.get_window_extent()
                    keep = lambda q: q[(q[:, 0] >= ab.x0) & (q[:, 0] <= ab.x1) & (q[:, 1] >= ab.y0) & (q[:, 1] <= ab.y1)]
                    pts, marks = keep(pts), keep(marks)
                grow = line.get_markersize() * fig.dpi / 72 / 2
                if _hits(bb, pts, 0.5 * line.get_linewidth() * fig.dpi / 72) or _hits(bb, marks, grow):
                    problems.append("%s '%s' touches a curve" % ("legend" if label == "legend" else "text", label))
                    break
            for cont in ax.containers:
                if isinstance(cont, BarContainer) and any(
                        _overlap(bb, Bbox.intersection(p.get_window_extent(), ax.get_window_extent()) or Bbox.null())
                        for p in cont.patches if p.get_visible()):
                    problems.append("text '%s' touches a bar" % label)
                    break
    # words placed inside a panel must not straddle its frame
    for label, bb, owner, art in boxes:
        if owner is not None and art in owner.texts and owner.axison and art.get_clip_on() is not None:
            ab = owner.get_window_extent()
            inside = ab.x0 + 1 < bb.x1 and bb.x0 < ab.x1 - 1 and ab.y0 + 1 < bb.y1 and bb.y0 < ab.y1 - 1
            if inside and (bb.x0 < ab.x0 - 1 or bb.x1 > ab.x1 + 1 or bb.y0 < ab.y0 - 1 or bb.y1 > ab.y1 + 1):
                problems.append("text '%s' crosses the panel frame" % label)
    # any text or tick label of one panel reaching into another panel
    for label, bb, owner, art in boxes:
        if owner is None:
            continue
        mine = {home(owner)} | {home(a) for a in owner.get_shared_x_axes().get_siblings(owner)
                                if a.get_window_extent().overlaps(owner.get_window_extent())}
        mine |= set(owner._twinned_axes.get_siblings(owner))
        for ax in main:
            if ax not in mine and _overlap(bb, ax.get_window_extent(), tol=2):
                problems.append("text '%s' reaches into another panel" % label)
    problems = sorted(set(problems))
    if not quiet:
        print("layout %-22s %s" % (name, "clean" if not problems else ""))
        for p in problems:
            print("    !", p)
    return problems




# ---------------------------------------------------------------------------
# report versions: drawn at Google Docs text width (6.5 in, letter, 1 in margins)
# ---------------------------------------------------------------------------
REPORT_W = 6.5
REPORT_FONT = 9


def report_halves(height, left=0.62, right=0.12, bottom=0.42, top=0.26):
    """Figure of width REPORT_W with one axes per half (identical boxes)."""
    setup(REPORT_FONT)
    fig = plt.figure(figsize=(REPORT_W, height))
    half = REPORT_W / 2
    w, h = (half - left - right) / REPORT_W, (height - bottom - top) / height
    boxes = [[(k * half + left) / REPORT_W, bottom / height, w, h] for k in (0, 1)]
    return fig, boxes


def report_single(height, width=None):
    setup(REPORT_FONT)
    return plt.figure(figsize=(width or REPORT_W, height))


def fit_report(fig, panels, gap=0.28, pad=0.03):
    """Move the axes so the visible content (tick labels, axis labels, titles,
    legends) of the first panel starts at the left edge of the figure, the
    content of the last ends at the right edge, every panel's content is equally
    wide, and all panels share one vertical band. panels: list of groups of
    axes; a group (e.g. two stacked axes) moves and stretches as one."""
    W, H = fig.get_size_inches()
    n = len(panels)
    cw = (W - 2 * pad - (n - 1) * gap) / n
    for _ in range(3):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        boxes = []
        for grp in panels:
            A = Bbox.union([ax.get_position() for ax in grp])
            C = Bbox.union([ax.get_tightbbox(r) for ax in grp]).transformed(inv)
            boxes.append((A, C))
        bot = max(A.y0 - C.y0 for A, C in boxes)
        top = max(C.y1 - A.y1 for A, C in boxes)
        y0, y1 = pad / H + bot, 1 - pad / H - top
        for k, (grp, (A, C)) in enumerate(zip(panels, boxes)):
            left = (pad + k * (cw + gap)) / W
            x0 = left + (A.x0 - C.x0)
            x1 = left + cw / W - (C.x1 - A.x1)
            for ax in grp:
                P = ax.get_position()
                fx0, fx1 = (P.x0 - A.x0) / A.width, (P.x1 - A.x0) / A.width
                fy0, fy1 = (P.y0 - A.y0) / A.height, (P.y1 - A.y0) / A.height
                ax.set_position([x0 + fx0 * (x1 - x0), y0 + fy0 * (y1 - y0),
                                 (fx1 - fx0) * (x1 - x0), (fy1 - fy0) * (y1 - y0)])




def save_report(fig, name, panels=None):
    """Save sims/figs/<name>.pdf and <name>.png at exactly the figure size.
    With panels, the axes are first fitted edge to edge (see fit_report)."""
    if panels:
        fit_report(fig, panels)
    probs = check_layout(fig, name)
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, name + ".pdf"))
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=300, facecolor="white")
    plt.close(fig)
    return probs


def band(ax, lo=800, hi=830, label=None, **kw):
    """Grey vertical (default) band, e.g. the trigger window."""
    ax.axvspan(lo, hi, color=BAND, zorder=0, lw=0, **kw)
