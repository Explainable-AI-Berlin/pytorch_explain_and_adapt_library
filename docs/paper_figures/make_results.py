import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

INK = "#1f2733"
MUT = "#6b7686"
RED = "#c0392b"
GREEN = "#2e8b57"
C_L = "#bcd3ee"
C_A = "#5f93cf"
C_V = "#22683f"
BG = "#f7f8fa"

# (name, latent, ambient, verified, is_confounder)   -- is_confounder = teacher verdict "false"
# Sources (2026-09-16): sweep_results.pt + direction_feedback.txt of each run dir.
data = {
    "Sparse Numbers\n(batch top-K SAE)": [
        ("Num128  ✓", 900, 39, 30, False),
        ("Num713  ✗", 453, 15, 2, True),
        ("Num797", 8, 1, 0, False),
        ("Num813", 13, 0, 0, False),
        ("Num757", 4, 1, 0, False),
    ],
    "NICO++\nCrocodile vs Lizard": [
        ("crocodile→lizard  ✓", 178, 41, 38, False),
        ("lizard→crocodile  ✓", 218, 31, 31, False),
        ("rocks→pasture  ✗", 35, 25, 23, True),
        ("vanuatu→rocks  ✗", 22, 15, 13, True),
        ("swinging→outdoors  ✗", 7, 6, 6, True),
    ],
    "ImageNet, natural\nFreight vs Passenger Car": [
        ("bin  ✓", 337, 72, 54, False),
        ("tracks  ✗", 288, 76, 40, True),
        ("graffiti  ✗", 85, 30, 22, True),
        ("travelling  ✓", 230, 41, 21, False),
    ],
    "ImageNet, planted jet\nFireboat vs Lifeboat": [
        ("squirting  ✗", 444, 109, 92, True),
        ("orange  ✓", 60, 18, 12, False),
        ("red  ✓", 50, 14, 9, False),
        ("fireworks  ✗", 122, 32, 4, True),
        ("hertfordshire  ✗", 309, 9, 4, True),
    ],
}
# AVERAGE group accuracy before -> after; gain = (after - before) / (1 - before), the
# normalised improvement CFKD logs as `gain`, printed in percent like the paper's tables. Third field = caption.
#   sparse numbers: only_sparse_numbers1k/.../classifier_poisoned100/didae (groups 0.668/0.975/0.995/0.330 -> 0.738/1.0/0.988/0.745)
#   celeba: didae_procrustes_free40_openai_clip_ddpm (0.645/0.949/0.953/0.436 -> 0.797/0.940/0.884/0.776); classical Procrustes CFKD 0.810
#   nico: openai_clip_rae_sde_g2_didae_msae_ep12, held-out 624 images (croc_grass/croc_rock/liz_grass/liz_rock
#         0.800/0.240/0.344/0.909 -> 0.900/0.336/0.477/0.859); the 60-image test split gives 0.667 -> 0.683
#   imagenet fireboat: fireboat_vs_lifeboat_curated5717_dinov3_linear/openai_clip_rae_guided_g2_single_jet_bs3_ep43s100_didae_msae
#         curated probe (every training fireboat carries a jet); AGA over the four (class x jet) groups on n=520
#         (0.972/0.940/1.000/0.909 -> 0.983/0.988/1.000/0.955); CFKD on the single ✗ direction #5717 with 26 counterfactuals;
#         all no-jet images 0.934 -> 0.981 (jet-free fireboats 0.940 -> 0.988, jet images 0.988 -> 0.993), overall test accuracy 0.973 -> 0.992
# CelebA Procrustes is withheld: the orthogonal-dictionary re-run ranks correctly but its
# CFKD repair is starved (40 counterfactuals vs 347) and regresses, so no honest tile exists yet.
# Means over the four groups above; 3-decimal group accuracies, so Gain is good to ~0.1.
def _aga(*g):
    return sum(g) / len(g)


gains = {
    0: (_aga(0.668, 0.975, 0.995, 0.330), _aga(0.738, 1.0, 0.988, 0.745), "CFKD on ✗ #713"),
    1: (_aga(0.800, 0.240, 0.344, 0.909), _aga(0.900, 0.336, 0.477, 0.859), "CFKD on ✗ dirs"),
    # freight car (natural probe), 2026-10-01 re-run of the tracks repair (the 2026-09-17 one never
    # trained): groups class x tracks atom #1661 active, n=520, 0.9643/0.9828/0.9412/0.9669 ->
    # 1.0/0.9742/0.9412/0.9711; overnight_group_eval.json of ..._ep40s100_didae_msae
    2: (_aga(0.9643, 0.9828, 0.9412, 0.9669), _aga(1.0, 0.9742, 0.9412, 0.9711), "CFKD on ✗ tracks"),
    3: (0.9553, 0.9814, "CFKD on ✗ dirs"),
}

# Authored at the final print width of an IEEEtran two-column figure* (7.16 in), so the
# figure is included at width=\textwidth with scale 1 and the font sizes below are the
# sizes that actually appear on the page. 2x2 rather than 1x4 for exactly that reason.
NC = 4
NR = 1
FS_T, FS_Y, FS_N, FS_X = 6.4, 5.6, 5.4, 5.4
fig = plt.figure(figsize=(7.16, 2.35))
gs = fig.add_gridspec(
    2,
    NC,
    height_ratios=[3.0, 1.3],
    hspace=0.3,
    wspace=0.62,
    left=0.085,
    right=0.99,
    top=0.80,
    bottom=0.13,
)
# No in-figure title: journal figures carry their explanation in the caption.

for idx, (title, rows) in enumerate(data.items()):
    r, col = divmod(idx, NC)
    gr = 0
    ax = fig.add_subplot(gs[gr, col])
    ax.set_facecolor(BG)
    names = [r[0] for r in rows]
    y = np.arange(len(rows))[::-1]
    xmax = max(max(r[1], r[2], r[3]) for r in rows)
    h = 0.26
    for r, yy in zip(rows, y):
        _, L, A, V, conf = r
        ax.barh(yy + h, L, height=h, color=C_L, zorder=3)
        ax.barh(yy, A, height=h, color=C_A, zorder=3)
        ax.barh(yy - h, V, height=h, color=C_V, zorder=3)
        ax.text(
            V + 0.03 * xmax,
            yy - h,
            ("%d" % V) if V > 0 else "0",
            va="center",
            ha="left",
            fontsize=FS_N,
            color=(C_V if V > 0 else MUT),
            fontweight="bold",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=FS_Y)
    for tick, r in zip(ax.get_yticklabels(), rows):
        if r[4]:
            tick.set_color(RED)
            tick.set_fontweight("bold")
    # linear bars without an x axis: the verified count printed beside its bar is the ranking key
    ax.set_xlim(0, xmax * 1.02)
    ax.set_xticks([])
    ax.set_title(title, fontsize=FS_T, fontweight="bold", linespacing=1.3, pad=4)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)

    axg = fig.add_subplot(gs[gr + 1, col])
    axg.set_facecolor(BG)
    axg.axis("off")
    b, a, src = gains[idx]
    axg.text(
        0.5,
        0.94,
        "average group accuracy",
        ha="center",
        va="top",
        fontsize=FS_X + 0.3,
        fontweight="bold",
        color=INK,
        transform=axg.transAxes,
    )
    if b is None:
        axg.text(
            0.5,
            0.42,
            src,
            ha="center",
            va="center",
            fontsize=FS_X,
            color=MUT,
            style="italic",
            transform=axg.transAxes,
            wrap=True,
        )
    else:
        up = a >= b
        after_c = GREEN if up else RED
        axg.barh(
            [0.62], [b], height=0.24, color="#d7a49a", transform=axg.transAxes, zorder=3
        )
        axg.barh(
            [0.30], [a], height=0.24, color=after_c, transform=axg.transAxes, zorder=3
        )
        axg.text(
            b + 0.02,
            0.62,
            f"before {b*100:.0f}%",
            va="center",
            fontsize=FS_X,
            color=INK,
            transform=axg.transAxes,
        )
        axg.text(
            a + 0.02,
            0.30,
            f"after {a*100:.0f}%",
            va="center",
            fontsize=FS_X,
            color=after_c,
            fontweight="bold",
            transform=axg.transAxes,
        )
        sign = "+" if up else "−"
        axg.text(
            0.5,
            0.13,
            f"Gain {sign}{100*abs(a-b)/(1-b):.1f} ({sign}{abs(a-b)*100:.1f} pts)\n{src}",
            ha="center",
            va="top",
            fontsize=FS_X - 0.7,
            color=after_c,
            fontweight="bold",
            transform=axg.transAxes,
        )
        axg.set_xlim(0, 1.0)

leg = [
    Patch(fc=C_L, label="latent flips (distilled probe)"),
    Patch(fc=C_A, label="ambient flips (real classifier)"),
    Patch(fc=C_V, label="verified flips (rank key)"),
    Patch(fc="white", ec=RED, label="confounder (labelled ✗)"),
]
fig.legend(
    handles=leg,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.995),
    ncol=4,
    fontsize=6.0,
    frameon=False,
    handlelength=1.4,
    columnspacing=1.4,
    handletextpad=0.5,
)

import os

out = os.environ.get(
    "FIG4_OUT",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "DiDAE_Journal_Paper",
        "figures",
        "figure_results_ranking_didae",
    ),
)
fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.08)
# PNG preview only, never beside the .pdf in the paper's figures/ dir (git churn)
prev = os.environ.get("FIG4_PNG")
if prev:
    matplotlib.use("agg")
    fig.savefig(prev, bbox_inches="tight", pad_inches=0.08, dpi=145)
print("saved %d-column figure -> %s.pdf" % (NC, out))
