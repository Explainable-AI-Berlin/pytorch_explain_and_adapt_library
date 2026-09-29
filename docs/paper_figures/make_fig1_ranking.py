"""
Figure 1 for the Ranking-DiDAE paper: one concrete Clever Hans of a DINOv3 linear probe
on ImageNet, found and ranked by DiDAE.  Default SPEC = the real fireboat-vs-lifeboat run
($PEAL_RUNS/imagenet_probes/fireboat_vs_lifeboat_dinov3_linear/openai_clip_rae_fresh_didae_msae,
2026-09-15): the probe's top directions are the "water jet" shortcut of Neuhaus et al.
(2022, arXiv:2212.04871, fireboat class).

Panels: (a) status quo, (b) ranking bar chart (latent / ambient / verified flips, log x),
(c) several counterfactuals per direction, row-aligned with (b), (d) the CFKD fix.

Any value given as a string in [brackets] is rendered amber as a placeholder; image
paths that do not exist become dashed slots.  Pass a JSON with the same keys as argv[1]
to override SPEC (paths in the JSON are taken relative to the JSON file).

usage:  python make_fig1_ranking.py                # fireboat (real numbers, fix pending)
        python make_fig1_ranking.py spec.json      # any other pair
"""

import sys, json, os
import matplotlib

matplotlib.use("pdf")
# Vector backends rasterise imshow() at 72 dpi and composite neighbouring images
# into one strip, which silently reduced the 256x256 counterfactuals to ~57x57 in
# the PDF. "none" lets the backend embed the original array (option_scale_image),
# and no compositing keeps every image its own full-resolution XObject.
matplotlib.rcParams["image.composite_image"] = False
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "fig1_assets_fireboat")


def _pairs(rank, k=3):
    return (
        [os.path.join(ASSETS, f"rank{rank}_{j}_orig.png") for j in range(k)],
        [os.path.join(ASSETS, f"rank{rank}_{j}_cf.png") for j in range(k)],
    )


# ------------------------------------------------------------------ SPEC (real fireboat run)
# Run: openai_clip_rae_guided_g2_didae_msae (2026-09-15 18:05-18:54): RAE decoder with DDPM
# edit-friendly inversion + classifier-free guidance scale 2 -> minimal edits of the input
# (PSNR 15.4 dB on the water-jet flips), pool 400+100, 313 candidates, 51 ambient / 31
# verified flips over 78 directions, edit realisation 0.24.
ASSETS = os.path.join(HERE, "fig1_assets_fireboat_g2")
SPEC = dict(
    title="",  # the message goes into the caption; leave empty
    method_label="Our pipeline (PEAL): rank every SAE direction by verified counterfactual flips → inspect the top ones → repair exactly those",
    status_quo_json=os.path.join(
        HERE, "topdataset_fireboat_vs_lifeboat", "top_dataset_atoms.json"
    ),
    d_box="pinpoint the exact shortcut\nrepair exactly that",
    n_dirs="6,144",
    which="which ones does the\nfireboat-vs-lifeboat probe use?",
    probe_line="DINOv3 ViT-L/16 linear probe, ImageNet fireboat vs lifeboat (99.2 % test acc.)\n"
    "public MSAE on CLIP ViT-L/14 (6,144 atoms), no SAE training\n"
    "500 images swept · 313 candidates rendered (DDPM inversion + CFG 2) · 78 directions",
    n_show=3,
    rows=[  # top-5 of 78 concept-replacement directions, sorted by verified flips (sweep_results.pt)
        dict(
            rank=1,
            name="add water jet",
            atoms="#1483 → #5717 squirting",
            flip="lifeboat → fireboat",
            verdict="spurious",
            latent=64,
            ambient=11,
            verified=10,
            orig=_pairs(1)[0],
            cf=_pairs(1)[1],
        ),
        dict(
            rank=2,
            name="red hull → white hull",
            atoms="#2882 red → #1483",
            flip="fireboat → lifeboat",
            verdict="unknown",
            latent=49,
            ambient=5,
            verified=5,
            orig=_pairs(2)[0],
            cf=_pairs(2)[1],
        ),
        dict(
            rank=3,
            name="coastal lifeboat livery",
            atoms="#2839 → #1223 cornwall",
            flip="fireboat → lifeboat",
            verdict="unknown",
            latent=10,
            ambient=3,
            verified=3,
            orig=_pairs(3)[0],
            cf=_pairs(3)[1],
        ),
        dict(
            rank=4,
            name="add spray plume",
            atoms="#240 orange → #3029 fireworks",
            flip="lifeboat → fireboat",
            verdict="spurious",
            latent=24,
            ambient=3,
            verified=2,
            orig=_pairs(4, 2)[0],
            cf=_pairs(4, 2)[1],
        ),
        dict(
            rank=5,
            name="remove jet, orange hull",
            atoms="#5717 squirting → #240 orange",
            flip="fireboat → lifeboat",
            verdict="spurious",
            latent=14,
            ambient=2,
            verified=2,
            orig=_pairs(5, 2)[0],
            cf=_pairs(5, 2)[1],
        ),
    ],
    verdict_header="harmful shortcut?\n(GT: Neuhaus et al.)",
    spurious_label="✗ water jet",
    unknown_label="? not in GT list",
    footer="ranks 1, 4 and 5 are the “water jet” shortcut of Neuhaus et al. (2022) for fireboat, rendered as minimal edits of the\n"
    "input: the same lifeboat scene gains a spraying jet and flips to fireboat at p = 0.99, a fireboat loses its jet and\n"
    "flips to lifeboat — found from the probe alone, with counterfactuals instead of highest-activating samples,\n"
    "and ranked among all directions in a single pass (all shown pairs of a direction when fewer than 3 verified flips exist)",
    # (d) qualitative fix evidence: validation images the ORIGINAL probe misclassified
    # because of the shortcut and the CFKD-finetuned probe gets right. Filled from
    # fig1_assets_fireboat_g2/fix_samples/samples.json (utils/find_confounder_fix_samples.py)
    # when that file exists; dashed placeholders otherwise.
    # (a) the status quo done for real: the 10 highest-activating training images of the
    # water-jet atom (utils-free: scratchpad top_activating.py -> topact/atom5717_top*.png)
    topact_dir=os.path.join(ASSETS, "topact"),
    topact_atom=5717,
    topact_caption="atom #5717 “squirting”: 10 highest-activating\ntraining images — all fireboats spraying",
    fix_samples_json=os.path.join(ASSETS, "fix_samples", "samples.json"),
    fix_n=2,
    class_names=["fireboat", "lifeboat"],
    out=os.path.join(HERE, "figure1_didae_fireboat"),
)
if len(sys.argv) > 1:
    j = json.load(open(sys.argv[1]))
    base = os.path.dirname(os.path.abspath(sys.argv[1]))
    for r in j.get("rows", []):
        for key in ("orig", "cf"):
            r[key] = [
                p if os.path.isabs(p) else os.path.join(base, p) for p in r.get(key, [])
            ]
    SPEC.update(j)

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
C_V = "#22683f"
FRAME = "#1f4e79"

W, H = 16.4, 7.6
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")


def panel(x0, y0, x1, y1, fc, ec, lw=1.4, z=0):
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            fc=fc,
            ec=ec,
            lw=lw,
            zorder=z,
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


def img_slot(cx, cy, s, path, label):
    if path and os.path.exists(path):
        ax.imshow(
            plt.imread(path),
            extent=(cx - s / 2, cx + s / 2, cy - s / 2, cy + s / 2),
            zorder=3,
            aspect="auto",
            interpolation="none",
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
BOT = 0.62  # bottom of the panels (room for the generality note below)
TOP = H - 0.75 if SPEC.get("method_label") else H - 0.25
Y0 = TOP - 1.05
STEP = 1.05
ROWY = [Y0 - i * STEP for i in range(N)]

# ================================================================= method frame around (b)-(d)
bx0, bx1 = 3.35, 7.55
cx0, cx1 = 7.75, 13.55
dx0, dx1 = 13.75, W - 0.2
if SPEC.get("method_label"):
    panel(bx0 - 0.10, BOT - 0.12, dx1 + 0.10, TOP + 0.34, "none", FRAME, lw=1.8, z=0)
    rbox(
        bx0 + 0.10,
        TOP + 0.06,
        dx1 - bx0 - 0.20,
        0.54,
        "white",
        FRAME,
        lw=1.0,
        rs=0.10,
        z=1,
    )
    txt(
        (bx0 + dx1) / 2,
        TOP + 0.33,
        SPEC["method_label"],
        fs=8.6,
        color=FRAME,
        bold=True,
        z=6,
    )

# ================================================================= (a) status quo
ax0, ax1 = 0.2, 3.15
panel(ax0, BOT, ax1, TOP, GREY_BG, GREY_ED)
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
gx0, gy0, gx1, gy1 = 0.42, 3.30, 2.93, 5.10
nc, nr = 30, 24
cw = (gx1 - gx0) / nc
ch = (gy1 - gy0) / nr
for r in range(nr):
    for c in range(nc):
        ax.add_patch(
            Rectangle(
                (gx0 + c * cw, gy0 + r * ch),
                cw * 0.8,
                ch * 0.8,
                fc="#dedbd5",
                ec="none",
                zorder=2,
            )
        )
ax.add_patch(
    Rectangle(
        (gx0, gy0), gx1 - gx0, gy1 - gy0, fill=False, ec=GREY_ED, lw=1.1, zorder=2
    )
)
mx, my = am, (gy0 + gy1) / 2
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
txt(am, gy1 + 0.30, f"{SPEC['n_dirs']} atoms — {SPEC['which']}", fs=8.0)
# highest-activating concept ON THE TRAINING SET, and its 10 highest-activating images
sq = None
if SPEC.get("status_quo_json") and os.path.exists(SPEC["status_quo_json"]):
    sq = json.load(open(SPEC["status_quo_json"]))[0]
ts = 0.44
gap = 0.06
ncol = 5
tx0 = am - (ncol * ts + (ncol - 1) * gap) / 2
ty0 = 1.62
tiles_top = ty0 + 2 * ts + gap
ax.add_line(
    Line2D(
        [mx, mx],
        [my - 0.40, tiles_top + 0.20],
        color=INK,
        lw=1.4,
        ls=(0, (2, 2)),
        zorder=5,
    )
)
arrow(mx, tiles_top + 0.22, mx, tiles_top + 0.04, lw=1.4, mut=9)
if sq:
    for k, pth in enumerate(sq["files"][:10]):
        rr, cc = divmod(k, ncol)
        cx = tx0 + cc * (ts + gap) + ts / 2
        cy = ty0 + (1 - rr) * (ts + gap) + ts / 2
        ax.imshow(
            plt.imread(pth),
            extent=(cx - ts / 2, cx + ts / 2, cy - ts / 2, cy + ts / 2),
            zorder=3,
            aspect="auto",
            interpolation="none",
        )
        ax.add_patch(
            Rectangle(
                (cx - ts / 2, cy - ts / 2),
                ts,
                ts,
                fill=False,
                ec="white",
                lw=0.8,
                zorder=4,
            )
        )
    txt(
        am,
        ty0 - 0.15,
        SPEC.get(
            "a_note",
            f"“{sq['name']}” (#{sq['atom']}): its 10 highest-activating images",
        ),
        fs=6.8,
        color=MUT,
        it=True,
    )
else:
    for k in range(10):
        rr, cc = divmod(k, ncol)
        ax.add_patch(
            Rectangle(
                (tx0 + cc * (ts + gap), ty0 + (1 - rr) * (ts + gap)),
                ts,
                ts,
                fc="#fff7ea",
                ec=PH,
                lw=0.8,
                ls=(0, (3, 2)),
                zorder=3,
            )
        )
    txt(
        am,
        ty0 - 0.15,
        "[highest-activating concept on the training set]",
        fs=6.6,
        color=PH,
        bold=True,
    )
rbox(0.40, BOT + 0.12, ax1 - ax0 - 0.40, 0.44, "#f6e3e1", RED, lw=1.0)
txt(
    am,
    BOT + 0.34,
    SPEC.get("a_verdict", "✖ is that the shortcut? unclear · no fix"),
    fs=8.2,
    color=RED,
    bold=True,
)

# ================================================================= (b) ranking: verified flips only
panel(bx0, BOT, bx1, TOP, BLUE_BG, BLUE_ED)
bm = (bx0 + bx1) / 2
txt(
    bm,
    TOP - 0.28,
    SPEC.get("b_title", "(b) Rank all directions by verified flips"),
    fs=11.5,
    bold=True,
)
if SPEC.get("probe_line"):
    txt(bm, TOP - 0.66, SPEC["probe_line"], fs=7.2, color=MUT, it=True)
BAR_X0, BAR_X1 = 5.75, 7.30
vmax = max(1, max(int(r["verified"]) for r in rows))
txt(
    (BAR_X0 + BAR_X1) / 2,
    ROWY[0] + STEP / 2 + 0.02,
    "verified counterfactual flips",
    fs=7.0,
    color=MUT,
    bold=True,
)
for r, yy in zip(rows, ROWY):
    V = int(r["verified"])
    wv = (BAR_X1 - BAR_X0) * V / vmax
    ax.add_patch(
        Rectangle((BAR_X0, yy - 0.13), max(wv, 0.05), 0.26, fc=C_V, ec="none", zorder=3)
    )
    txt(
        BAR_X0 + max(wv, 0.05) + 0.08,
        yy,
        str(V),
        fs=8.5,
        color=C_V,
        bold=True,
        ha="left",
    )
    spur = r["verdict"] == "spurious"
    ax.add_patch(
        Circle((bx0 + 0.30, yy), 0.17, fc=(RED if spur else BLUE), ec="none", zorder=4)
    )
    txt(bx0 + 0.30, yy, str(r["rank"]), fs=9.5, color="white", bold=True)
    txt(
        bx0 + 0.56,
        yy + 0.19,
        r["name"],
        fs=7.9,
        bold=True,
        color=(RED if spur else INK),
        ha="left",
    )
    txt(bx0 + 0.56, yy - 0.02, r["flip"], fs=7.0, color=INK, ha="left")
    txt(bx0 + 0.56, yy - 0.21, r["atoms"], fs=6.0, color=MUT, ha="left")
if SPEC.get("b_note"):
    txt(bm, ROWY[-1] - STEP / 2 - 0.30, SPEC["b_note"], fs=6.8, color=MUT, it=True)

# ================================================================= (c) counterfactuals
panel(cx0, BOT, cx1, TOP, BLUE_BG, BLUE_ED)
cm = (cx0 + cx1) / 2
txt(
    cm,
    TOP - 0.28,
    SPEC.get("c_title", f"(c) Counterfactuals of the top-{N} only"),
    fs=11.5,
    bold=True,
)
if SPEC.get("c_subtitle"):
    txt(cm, TOP - 0.56, SPEC["c_subtitle"], fs=8.2, color=MUT, it=True)
IMG = 0.58
PAIR_W = 2 * IMG + 0.24
PAD = 0.12
X_TEACH = cx1 - 0.55
gx = cx0 + 0.10
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
        if j >= len(r["orig"]):
            continue
        o = r["orig"][j]
        c = r["cf"][j] if j < len(r["cf"]) else None
        img_slot(px + IMG / 2, yy, IMG, o, "orig.")
        arrow(px + IMG + 0.02, yy, px + IMG + 0.22, yy)
        img_slot(px + IMG + 0.24 + IMG / 2, yy, IMG, c, "counter-\nfactual")
    if spur:
        rbox(X_TEACH - 0.46, yy - 0.16, 0.92, 0.32, "#f6e3e1", RED, lw=1.0)
        txt(X_TEACH, yy, SPEC["spurious_label"], fs=7.6, color=RED, bold=True)
    elif r["verdict"] == "ok":
        rbox(X_TEACH - 0.46, yy - 0.16, 0.92, 0.32, GREENBG, GREEN, lw=1.0)
        txt(
            X_TEACH,
            yy,
            SPEC.get("ok_label", "✓ class evidence"),
            fs=7.0,
            color=GREEN,
            bold=True,
        )
    else:
        rbox(X_TEACH - 0.46, yy - 0.16, 0.92, 0.32, "#f4f4f2", GREY_ED, lw=1.0)
        txt(X_TEACH, yy, SPEC["unknown_label"], fs=7.0, color=MUT, bold=True)
txt(
    X_TEACH,
    ROWY[0] + STEP / 2 + 0.05,
    SPEC["verdict_header"],
    fs=6.8,
    color=MUT,
    bold=True,
)
if SPEC.get("footer"):
    txt(cm, ROWY[-1] - STEP / 2 - 0.55, SPEC["footer"], fs=6.3, color=MUT, it=True)

# ================================================================= (d) fix
panel(dx0, BOT, dx1, TOP, BLUE_BG, BLUE_ED)
dm = (dx0 + dx1) / 2
txt(dm, TOP - 0.28, SPEC.get("d_title", "(d) Fix the probe"), fs=10.6, bold=True)
if SPEC.get("fix_subtitle"):
    txt(dm, TOP - 0.58, SPEC["fix_subtitle"], fs=7.4, color=MUT, it=True)
samples = []
if os.path.exists(SPEC["fix_samples_json"]):
    samples = json.load(open(SPEC["fix_samples_json"]))[: int(SPEC["fix_n"])]
cn = SPEC["class_names"]
SY0 = TOP - 1.50
SSTEP = 2.15
SIMG = 0.90
GAPX = 0.30
for i in range(int(SPEC["fix_n"])):
    yy = SY0 - i * SSTEP
    smp = samples[i] if i < len(samples) else None
    xo = dm - SIMG / 2 - GAPX / 2
    xc = dm + SIMG / 2 + GAPX / 2
    if smp and not smp.get("cf_image"):
        SN = 1.0
        img_slot(dm, yy, SN, smp["image"], "validation\nimage")
        pb = smp["p_before"]
        pa = smp["p_after"]
        kb = int(max(range(2), key=lambda k: pb[k]))
        ka = int(max(range(2), key=lambda k: pa[k]))
        txt(
            dm,
            yy + SN / 2 + 0.10,
            smp.get("caption", "natural held-out image"),
            fs=6.4,
            color=MUT,
        )
        txt(
            dm,
            yy - SN / 2 - 0.13,
            f"before: {cn[kb]}  p={pb[kb]:.2f}",
            fs=7.0,
            color=RED,
            bold=True,
        )
        txt(
            dm,
            yy - SN / 2 - 0.30,
            f"after:  {cn[ka]}  p={pa[ka]:.2f}",
            fs=7.0,
            color=GREEN,
            bold=True,
        )
        txt(dm, yy - SN / 2 - 0.46, f"true: {cn[smp['true']]}", fs=6.6, color=MUT)
        continue
    # counterfactual only: the edited image and the decision it flips
    CN = 1.18
    img_slot(dm, yy, CN, (smp.get("cf_image") if smp else None), "counter-\nfactual")
    if smp:
        pb = smp["p_before"]
        pa = smp["p_after"]
        kb = int(max(range(2), key=lambda k: pb[k]))
        ka = int(max(range(2), key=lambda k: pa[k]))
        txt(
            dm,
            yy + CN / 2 + 0.11,
            "counterfactual: + "
            + (smp.get("caption") or SPEC["spurious_label"].lstrip("\u2717 ")),
            fs=6.8,
            color=MUT,
        )
        txt(
            dm,
            yy - CN / 2 - 0.15,
            f"before the fix:  {cn[kb]}  ✗",
            fs=8.0,
            color=RED,
            bold=True,
        )
        txt(
            dm,
            yy - CN / 2 - 0.34,
            f"after the fix:  {cn[ka]}  ✓",
            fs=8.0,
            color=GREEN,
            bold=True,
        )

    else:
        txt(dm, yy - 0.75, f"before: [{cn[0]}  p=0.__]", fs=7.0, color=PH, bold=True)
        txt(dm, yy - 0.92, f"after:  [{cn[1]}  p=0.__]", fs=7.0, color=PH, bold=True)
if samples:
    if SPEC.get("fix_footer"):
        txt(dm, BOT + 0.82, SPEC["fix_footer"], fs=6.0, color=MUT, it=True)
else:
    txt(dm, BOT + 0.82, "CFKD run pending", fs=7.2, color=PH, it=True, bold=True)
for _x0, _x1, _key, _default in (
    (bx0, bx1, "b_box", ""),
    (cx0, cx1, "c_box", ""),
    (dx0, dx1, "d_box", ""),
):
    _t = SPEC.get(_key, _default)
    if not _t:
        continue
    rbox(_x0 + 0.15, BOT + 0.12, _x1 - _x0 - 0.30, 0.52, "white", FRAME, lw=1.2)
    txt((_x0 + _x1) / 2, BOT + 0.38, _t, fs=8.0, color=FRAME, bold=True)

fig.savefig(SPEC["out"] + ".pdf")
fig.savefig(SPEC["out"] + ".png", dpi=440)
print("wrote", SPEC["out"] + ".{pdf,png}")
