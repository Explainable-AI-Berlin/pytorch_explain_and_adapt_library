"""
Template Figure 1 for the Ranking-DiDAE paper, anchored on the headline Clever Hans of
Neuhaus et al. (2022, arXiv:2212.04871): ImageNet class "hummingbird" <-> "bird feeder".

Panels: (a) status quo, (b) ranking bar chart (latent / ambient / verified flips, log x,
same style as the results figure), (c) several counterfactuals per direction, row-aligned
with (b), (d) the CFKD fix.

PLACEHOLDER=True renders every number in amber [brackets] and every missing image as a
dashed slot. To fill the figure: edit SPEC or pass a JSON with the same keys as argv[1]
(image entries are lists of paths, one per shown counterfactual), set "placeholder": false.

usage:  python make_fig1_hummingbird.py            # template
        python make_fig1_hummingbird.py spec.json  # filled
"""

import sys, json, os
import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
from matplotlib.lines import Line2D
import numpy as np

# ------------------------------------------------------------------ SPEC (fill me)
SPEC = dict(
    placeholder=True,
    probe="DINOv3 linear probe, ImageNet class “hummingbird”",
    n_dirs="10,240",  # dictionary size of the (public) SAE
    n_sweep=512,  # images swept per direction
    n_show=3,  # counterfactual pairs shown per direction
    # top-5 directions by verified ambient flips.  Names / verdicts are hypotheses until
    # the run finishes: rank 1 & 2 are the Neuhaus et al. Clever Hans cues, 3-5 plausible
    # class features.  Counts are illustrative.  orig / cf: lists of image paths.
    rows=[
        dict(
            rank=1,
            dim="#____",
            name="bird feeder",
            verdict="spurious",
            latent=391,
            ambient=274,
            verified=231,
            orig=[],
            cf=[],
        ),
        dict(
            rank=2,
            dim="#____",
            name="red flowers",
            verdict="spurious",
            latent=350,
            ambient=240,
            verified=198,
            orig=[],
            cf=[],
        ),
        dict(
            rank=3,
            dim="#____",
            name="long thin beak",
            verdict="valid",
            latent=330,
            ambient=215,
            verified=180,
            orig=[],
            cf=[],
        ),
        dict(
            rank=4,
            dim="#____",
            name="iridescent plumage",
            verdict="valid",
            latent=300,
            ambient=190,
            verified=151,
            orig=[],
            cf=[],
        ),
        dict(
            rank=5,
            dim="#____",
            name="hovering wing blur",
            verdict="valid",
            latent=280,
            ambient=160,
            verified=122,
            orig=[],
            cf=[],
        ),
    ],
    # fix: CFKD on the teacher-labelled counterfactuals
    fix_metric="p(hummingbird) on\nfeeder-only images",
    fix_before=0.57,
    fix_after=0.13,  # SpuFix numbers as stand-in
    fix_metric2="worst-group accuracy",
    fix2_before=0.61,
    fix2_after=0.79,
    out=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "figure1_didae_hummingbird"
    ),
)
if len(sys.argv) > 1:
    SPEC.update(json.load(open(sys.argv[1])))
PLH = SPEC["placeholder"]

# ------------------------------------------------------------------ palette / helpers
INK = "#1f2733"
MUT = "#6b7686"
PH = "#d98a1f"
GREY_BG = "#f2f1ee"
GREY_ED = "#c9c6bf"
RED = "#c0392b"
REDBG = "#fbeae8"
BLUE_BG = "#eaf2fb"
BLUE_ED = "#9dc0e6"
BLUE = "#2f6db4"
GREEN = "#2e8b57"
GREENBG = "#e7f4ec"
C_L = "#bcd3ee"
C_A = "#5f93cf"
C_V = "#22683f"

W, H = 16.4, 7.3
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")


def panel(x0, y0, x1, y1, fc, ec):
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            fc=fc,
            ec=ec,
            lw=1.4,
            zorder=0,
        )
    )


def rbox(x0, y0, w, h, fc, ec, lw=1.2, rs=0.08, z=3):
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            w,
            h,
            boxstyle=f"round,pad=0.01,rounding_size={rs}",
            fc=fc,
            ec=ec,
            lw=lw,
            zorder=z,
        )
    )


def arrow(x0, y0, x1, y1, color=INK, lw=1.6, mut=11):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=mut,
            lw=lw,
            color=color,
            shrinkA=1,
            shrinkB=1,
            zorder=4,
        )
    )


def txt(x, y, s, fs=10, color=INK, ha="center", va="center", bold=False, it=False, z=5):
    ax.text(
        x,
        y,
        s,
        ha=ha,
        va=va,
        fontsize=fs,
        color=color,
        zorder=z,
        linespacing=1.15,
        fontweight="bold" if bold else "normal",
        fontstyle="italic" if it else "normal",
    )


def num(v, fmt="{}"):
    """format a number; amber [bracketed] while the figure is a template"""
    s = fmt.format(v)
    return (f"[{s}]", PH, True) if PLH else (s, INK, False)


def img_slot(cx, cy, s, path, label):
    if path and os.path.exists(path):
        ax.imshow(
            plt.imread(path),
            extent=(cx - s / 2, cx + s / 2, cy - s / 2, cy + s / 2),
            zorder=3,
            aspect="auto",
        )
        ax.add_patch(
            Rectangle(
                (cx - s / 2, cy - s / 2), s, s, fill=False, ec=INK, lw=0.7, zorder=4
            )
        )
    else:
        ax.add_patch(
            Rectangle(
                (cx - s / 2, cy - s / 2),
                s,
                s,
                fc="#fff7ea",
                ec=PH,
                lw=1.0,
                ls=(0, (3, 2)),
                zorder=3,
            )
        )
        txt(cx, cy, label, fs=6.6, color=PH, bold=True)


rows = SPEC["rows"]
N = len(rows)
K = int(SPEC["n_show"])
TOP = H - 0.75  # top of all panels
Y0 = TOP - 1.30  # centre of first row
STEP = 0.82  # row pitch
ROWY = [Y0 - i * STEP for i in range(N)]

# ================================================================= title
txt(
    W / 2,
    H - 0.33,
    "From 10K unlabeled SAE dimensions to the five that matter for one probe — ranked by verified counterfactual flips, then fixed",
    fs=13,
    bold=True,
)

# ================================================================= (a) status quo
ax0, ax1 = 0.2, 3.15
panel(ax0, 0.25, ax1, TOP, GREY_BG, GREY_ED)
am = (ax0 + ax1) / 2
txt(am, TOP - 0.28, "(a) Today: browse an SAE", fs=11.5, bold=True)
txt(
    am,
    TOP - 0.56,
    "manual · correlational · not actionable",
    fs=8.4,
    color=MUT,
    it=True,
)
gx0, gy0, gx1, gy1 = 0.42, 3.0, 2.93, 5.35
nc, nr = 30, 26
cw = (gx1 - gx0) / nc
ch = (gy1 - gy0) / nr
rng = np.random.default_rng(3)
hi = set((int(rng.integers(nr)), int(rng.integers(nc))) for _ in range(9))
for r in range(nr):
    for c in range(nc):
        ax.add_patch(
            Rectangle(
                (gx0 + c * cw, gy0 + r * ch),
                cw * 0.8,
                ch * 0.8,
                fc=(PH if (r, c) in hi else "#dedbd5"),
                ec="none",
                zorder=2,
            )
        )
ax.add_patch(
    Rectangle(
        (gx0, gy0), gx1 - gx0, gy1 - gy0, fill=False, ec=GREY_ED, lw=1.1, zorder=2
    )
)
mx, my = am, 4.15
ax.add_patch(Circle((mx, my), 0.40, fill=False, ec=INK, lw=2.2, zorder=6))
ax.add_line(
    Line2D(
        [mx + 0.29, mx + 0.68],
        [my - 0.29, my - 0.68],
        color=INK,
        lw=3.0,
        zorder=6,
        solid_capstyle="round",
    )
)
txt(
    am,
    2.72,
    f"{SPEC['n_dirs']} dims — which ones does\nthe hummingbird probe use?",
    fs=8.4,
)
txt(am, 2.28, "per dim: highest-activating images", fs=8.2, bold=True)
tx0, ty0 = 0.50, 1.36
ts = 0.35
gap = 0.09
cols = [
    "#8fb0d6",
    "#b7cbe6",
    "#a9c4a0",
    "#d9c48f",
    "#c9a9c5",
    "#9fc7c7",
    "#d6a58f",
    "#a0b8c9",
    "#c3c9a0",
    "#b8a9d9",
    "#9fc0a5",
    "#d9b7a0",
]
k = 0
for rr in range(2):
    for cc in range(6):
        ax.add_patch(
            Rectangle(
                (tx0 + cc * (ts + gap), ty0 + rr * (ts + gap)),
                ts,
                ts,
                fc=cols[k % 12],
                ec="white",
                lw=0.8,
                zorder=3,
            )
        )
        k += 1
rbox(0.40, 0.42, ax1 - ax0 - 0.40, 0.44, "#f6e3e1", RED, lw=1.0)
txt(am, 0.64, "✖ feeder or bird? unclear · no fix", fs=8.2, color=RED, bold=True)

# ================================================================= (b) ranking bar chart
bx0, bx1 = 3.35, 7.55
panel(bx0, 0.25, bx1, TOP, BLUE_BG, BLUE_ED)
bm = (bx0 + bx1) / 2
txt(bm, TOP - 0.28, "(b) Rank all directions by verified flips", fs=11.5, bold=True)
txt(
    bm,
    TOP - 0.56,
    f"{SPEC['probe']}\npublic SAE, no SAE training · {SPEC['n_sweep']} images swept per direction",
    fs=8.2,
    color=MUT,
    it=True,
)
# real axes in inch coordinates so bars align with the image rows of (c)
BAR_X0, BAR_X1 = 5.35, 7.40
BY0, BY1 = ROWY[-1] - STEP / 2 - 0.02, ROWY[0] + STEP / 2
axb = fig.add_axes([BAR_X0 / W, BY0 / H, (BAR_X1 - BAR_X0) / W, (BY1 - BY0) / H])
axb.set_ylim(BY0, BY1)
axb.patch.set_alpha(0)
axb.set_xscale("log")
axb.set_xlim(0.7, 3000)
h = 0.22
for r, yy in zip(rows, ROWY):
    L, A, V = r["latent"], r["ambient"], r["verified"]
    axb.barh(yy + h, max(L, 0.7), height=h, color=C_L, zorder=3)
    axb.barh(yy, max(A, 0.7), height=h, color=C_A, zorder=3)
    axb.barh(yy - h, max(V, 0.7), height=h, color=C_V, zorder=3)
    s, c, b = num(V)
    axb.text(
        max(V, 0.7) * 1.25,
        yy - h,
        s,
        va="center",
        ha="left",
        fontsize=8,
        color=(PH if PLH else C_V),
        fontweight="bold",
    )
axb.set_yticks([])
axb.tick_params(axis="x", labelsize=7.5, colors=MUT)
for sp in ("top", "right", "left"):
    axb.spines[sp].set_visible(False)
axb.spines["bottom"].set_color(BLUE_ED)
axb.grid(axis="x", ls=":", color="#c5d3e6", zorder=0)
axb.set_xlabel("flips  (log scale)", fontsize=8, color=MUT, labelpad=2)
axb.xaxis.set_label_coords(0.5, -0.36 / (BY1 - BY0))
# rank badges + direction names as y labels (drawn on the main canvas)
for r, yy in zip(rows, ROWY):
    spur = r["verdict"] == "spurious"
    ax.add_patch(
        Circle((bx0 + 0.32, yy), 0.17, fc=(RED if spur else BLUE), ec="none", zorder=4)
    )
    txt(bx0 + 0.32, yy, str(r["rank"]), fs=9.5, color="white", bold=True)
    txt(
        bx0 + 0.60,
        yy + 0.10,
        r["name"],
        fs=8.2,
        bold=True,
        color=(RED if spur else INK),
        ha="left",
    )
    s, c, b = num(r["dim"])
    txt(
        bx0 + 0.60,
        yy - 0.14,
        f"dim {s}",
        fs=7.0,
        color=(PH if PLH else MUT),
        ha="left",
        bold=b,
    )
# legend
ly = ROWY[-1] - STEP / 2 - 0.78
for i, (c, l) in enumerate(
    [
        (C_L, "latent flips (distilled probe)"),
        (C_A, "ambient flips (real classifier)"),
        (C_V, "verified flips — rank key"),
    ]
):
    ax.add_patch(
        Rectangle(
            (bx0 + 0.30, ly - i * 0.22 - 0.07), 0.24, 0.14, fc=c, ec="none", zorder=3
        )
    )
    txt(bx0 + 0.62, ly - i * 0.22, l, fs=7.8, ha="left", color=INK)

# ================================================================= (c) counterfactuals
cx0, cx1 = 7.75, 13.55
panel(cx0, 0.25, cx1, TOP, BLUE_BG, BLUE_ED)
cm = (cx0 + cx1) / 2
txt(
    cm,
    TOP - 0.28,
    f"(c) Counterfactuals of the top-5 only ({K} verified flips per direction)",
    fs=11.5,
    bold=True,
)
txt(
    cm,
    TOP - 0.56,
    "original $\\mathbf{x}$  →  counterfactual $\\tilde{\\mathbf{x}} = $ decode$(\\mathbf{z} + \\delta\\,\\mathbf{v}_k)$ — the probe's decision flips on every pair",
    fs=8.2,
    color=MUT,
    it=True,
)
IMG = 0.56
PAIR_W = 2 * IMG + 0.24
PAD = 0.12
grid_w = K * PAIR_W + (K - 1) * PAD
X_TEACH = cx1 - 0.62
gx = cx0 + 0.18
for r, yy in zip(rows, ROWY):
    spur = r["verdict"] == "spurious"
    if spur:
        rbox(
            cx0 + 0.10,
            yy - STEP / 2 + 0.05,
            cx1 - cx0 - 0.20,
            STEP - 0.10,
            REDBG,
            RED,
            lw=0.9,
            rs=0.10,
            z=1,
        )
    for j in range(K):
        px = gx + j * (PAIR_W + PAD)
        o = r["orig"][j] if j < len(r["orig"]) else None
        c = r["cf"][j] if j < len(r["cf"]) else None
        img_slot(px + IMG / 2, yy, IMG, o, "orig.")
        arrow(px + IMG + 0.02, yy, px + IMG + 0.22, yy)
        img_slot(px + IMG + 0.24 + IMG / 2, yy, IMG, c, "counter-\nfactual")
    if spur:
        rbox(X_TEACH - 0.44, yy - 0.16, 0.88, 0.32, "#f6e3e1", RED, lw=1.0)
        txt(X_TEACH, yy, "✗ spurious", fs=7.8, color=RED, bold=True)
    else:
        rbox(X_TEACH - 0.44, yy - 0.16, 0.88, 0.32, GREENBG, GREEN, lw=1.0)
        txt(X_TEACH, yy, "✓ valid", fs=7.8, color=GREEN, bold=True)
txt(
    X_TEACH, ROWY[0] + STEP / 2 + 0.06, "teacher\nverdict", fs=7.8, color=MUT, bold=True
)
txt(
    cm,
    ROWY[-1] - STEP / 2 - 0.42,
    "ranks 1–2 reproduce the “bird feeder” / “red flowers” Clever Hans of Neuhaus et al. (2022) for hummingbird —\n"
    "found from the probe alone, with precise counterfactuals instead of highest-activating samples,\n"
    "and ranked among all directions in a single pass",
    fs=7.2,
    color=MUT,
    it=True,
)

# ================================================================= (d) fix
dx0, dx1 = 13.75, W - 0.2
panel(dx0, 0.25, dx1, TOP, GREENBG, GREEN)
dm = (dx0 + dx1) / 2
txt(dm, TOP - 0.28, "(d) Fix the probe", fs=11.5, bold=True)
txt(
    dm,
    TOP - 0.56,
    "counterfactual knowledge\ndistillation (CFKD)",
    fs=8.4,
    color=MUT,
    it=True,
)
rbox(dx0 + 0.15, 4.55, dx1 - dx0 - 0.30, 0.62, "white", GREY_ED)
txt(
    dm,
    4.86,
    "teacher verdicts label the\ncounterfactuals: ✗ must not\nchange the prediction",
    fs=7.8,
)
arrow(dm, 4.52, dm, 4.18, color=INK)
rbox(dx0 + 0.15, 3.55, dx1 - dx0 - 0.30, 0.60, "white", GREY_ED)
txt(dm, 3.85, "distill probe on real +\ncounterfactual images → $\\mathbf{w}'$", fs=7.8)


def bar_pair(yb, label, b, a, pct):
    txt(dm, yb + 0.42, label, fs=8.0, bold=True)
    for k, (v, c, tag) in enumerate([(b, RED, "before"), (a, GREEN, "after")]):
        yy = yb + 0.12 - k * 0.24
        f = max(float(v), 0.02)
        ax.add_patch(
            Rectangle(
                (dx0 + 0.78, yy - 0.08), 1.25 * f, 0.16, fc=c, ec="none", zorder=3
            )
        )
        txt(dx0 + 0.72, yy, tag, fs=7, color=MUT, ha="right")
        s, col, bd = num(v, "{:.0%}" if pct else "{:.2f}")
        txt(dx0 + 0.78 + 1.25 * f + 0.05, yy, s, fs=7.4, ha="left", color=col, bold=bd)


bar_pair(2.55, SPEC["fix_metric"], SPEC["fix_before"], SPEC["fix_after"], False)
bar_pair(1.45, SPEC["fix_metric2"], SPEC["fix2_before"], SPEC["fix2_after"], True)
rbox(dx0 + 0.15, 0.42, dx1 - dx0 - 0.30, 0.44, "white", GREEN, lw=1.0)
txt(dm, 0.64, "✓ backbone untouched", fs=8.2, color=GREEN, bold=True)

if PLH:
    txt(
        W / 2,
        0.09,
        "template: amber [brackets], illustrative bar lengths and dashed slots are placeholders to be filled from the finished DINOv3 run",
        fs=7.2,
        color=PH,
        it=True,
    )

fig.savefig(SPEC["out"] + ".pdf")
fig.savefig(SPEC["out"] + ".png", dpi=105)
print("wrote", SPEC["out"] + ".{pdf,png}")
