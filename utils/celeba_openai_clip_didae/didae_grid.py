"""DiDAE discovery-half parameter grid on the CelebA OpenAI-CLIP generator.

For each variant: write a config derived from a base yaml, copy the shared distilled
probe into its base_dir (overwrite False -> same boundary in every run), run
run_didae.py, then summarise sweep_results.pt + direction_feedback.txt + log lines
into grid_summary.md.   Usage: didae_grid.py <variants: all | name,name,...>
"""

import os, sys, re, subprocess, shutil, time, json
import yaml, torch

S = os.environ.get(
    "DIDAE_GRID_OUT", os.path.dirname(os.path.abspath(__file__))
)  # where configs, logs and grid_summary.md land
BASE = os.environ["PEAL_BASE"]
RUNS = os.environ["PEAL_RUNS"]
CFG_DIR = f"{BASE}/configs/didae_experiments/adaptors"
RUN_ROOT = f"{RUNS}/celeba1k/Blond_Hair/classifier_poisoned098"
PROBE_SRC = f"{RUN_ROOT}/didae_msae_openai_clip_ddpm/distilled_probe"
DDPM = "<PEAL_BASE>/configs/didae_experiments/explainers/dae_distill_ddpm.yaml"
DDIM = "<PEAL_BASE>/configs/didae_experiments/explainers/dae_distill.yaml"
MSAE_BASE = f"{CFG_DIR}/celeba1kx098_resnet18_didae_sae_vit_l14_ddim.yaml"
PROC_BASE = f"{CFG_DIR}/celeba1kx098_resnet18_didae_procrustes_openai_clip.yaml"

# name -> (base config, overrides)
V = {}


def add(name, base, sampler, **kw):
    ov = {
        "explainer": DDPM if sampler == "ddpm" else DDIM,
        "base_dir": f"$PEAL_RUNS/celeba1k/Blond_Hair/classifier_poisoned098/grid_{name}",
        "overwrite": False,
        "run_cfkd_on_false_directions": False,
    }
    ov.update(kw)
    V[name] = (base, ov)


for dic, base in (("msae", MSAE_BASE), ("proc", PROC_BASE)):
    for smp in ("ddpm", "ddim"):
        add(f"{dic}_{smp}_b2", base, smp, component_bounds_scale=2.0)
        add(f"{dic}_{smp}_b3", base, smp, component_bounds_scale=3.0)
        add(f"{dic}_{smp}_random", base, smp, decode_selection="random")
        add(
            f"{dic}_{smp}_repl",
            base,
            smp,
            concept_replacement=True,
            concept_replacement_unique=True,
            concept_replacement_unique_strategy="greedy",
            concept_replacement_candidates=256 if dic == "msae" else 40,
        )

# --- grid 2: around the winners ---
add("msae_ddpm_b4", MSAE_BASE, "ddpm", component_bounds_scale=4.0)
add(
    "msae_ddpm_b3_topk20",
    MSAE_BASE,
    "ddpm",
    component_bounds_scale=3.0,
    top_k_directions=20,
)
add(
    "msae_ddpm_b3_repl",
    MSAE_BASE,
    "ddpm",
    component_bounds_scale=3.0,
    concept_replacement=True,
    concept_replacement_unique=True,
    concept_replacement_unique_strategy="greedy",
    concept_replacement_candidates=256,
)
add(
    "msae_ddpm_b3_lasso02",
    MSAE_BASE,
    "ddpm",
    component_bounds_scale=3.0,
    lasso_alpha=0.02,
    overwrite=True,
)
add(
    "msae_ddpm_b3_lasso0",
    MSAE_BASE,
    "ddpm",
    component_bounds_scale=3.0,
    lasso_alpha=0.0,
    overwrite=True,
)
add(
    "proc_ddpm_b2_topk20",
    PROC_BASE,
    "ddpm",
    component_bounds_scale=2.0,
    top_k_directions=20,
)
add("proc_ddpm_b15", PROC_BASE, "ddpm", component_bounds_scale=1.5)


def run(name):
    base, ov = V[name]
    cfg = yaml.safe_load(open(base))
    cfg.update(ov)
    path = f"{S}/grid_cfg_{name}.yaml"
    yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)
    bd = os.path.expandvars(cfg["base_dir"])
    os.makedirs(bd, exist_ok=True)
    if not os.path.exists(f"{bd}/distilled_probe/w.pt"):
        shutil.copytree(PROBE_SRC, f"{bd}/distilled_probe", dirs_exist_ok=True)
    log = f"{S}/grid_{name}.log"
    t0 = time.time()
    with open(log, "w") as f:
        # the driver itself runs inside the container, so call python directly
        rc = subprocess.call(
            [sys.executable, "-W", "ignore", "run_didae.py", "--config", path],
            cwd=BASE,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    return rc, log, bd, time.time() - t0


def summarise(name, rc, log, bd, dt):
    txt = open(log, errors="replace").read()

    def grab(pat, default="-"):
        m = re.search(pat, txt)
        return m.group(1) if m else default

    row = {
        "variant": name,
        "rc": rc,
        "min": f"{dt/60:.1f}",
        "latent": grab(r"(\d+) total latent flips"),
        "decoded": grab(r"Decoding (\d+) counterfactual candidates"),
        "ambient": grab(r"(\d+)/\d+ rendered images flipped ambient"),
        "verified": grab(r"(\d+)/\d+ rendered images flipped predictor AND"),
        "realised": grab(r"Edit realisation.*?median ([0-9.]+)"),
        "specificity": grab(r"Avg Specificity: ([0-9.]+)"),
    }
    male = "-"
    true_d = []
    false_d = []
    sr = f"{bd}/sweep_results.pt"
    names = {}
    if os.path.exists(sr):
        for e in torch.load(sr, map_location="cpu", weights_only=False):
            names[e["direction_idx"]] = e["dimension_name"]
            n = e["dimension_name"]
            if (
                ("Male" in n and "Male [" in n)
                or "homme" in n
                or n.startswith("Male")
                or "#4898" in n
                or n.startswith("OFF #4898")
                or "ON #4898" in n
                or "-> ON #20 " in n
            ):
                male += f" | {n[:50]}: lat {e['latent_flip_count']} tried {e['total_attempted']} amb {e['ambient_flip_count']} ver {e['success_count']}"
    fb = f"{bd}/direction_feedback.txt"
    if os.path.exists(fb):
        for line in open(fb):
            m = re.match(
                r"direction=(\d+), feedback=(\w+), success_count=(\d+), n_true=(\d+), n_false=(\d+)",
                line,
            )
            if m:
                d, f, s, nt, nf = m.groups()
                tag = f"{names.get(int(d), d)[:28]}({nt}/{nf})"
                (true_d if f == "true" else false_d).append(tag)
    row["male"] = male.lstrip(" -|") or "-"
    row["true_dirs"] = "; ".join(true_d) or "-"
    row["false_dirs"] = "; ".join(false_d) or "-"
    tb = re.search(r"Traceback[\s\S]{0,600}", txt)
    row["error"] = tb.group(0).splitlines()[-1][:120] if tb else ""
    return row


if __name__ == "__main__":
    sel = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(V) if sel == "all" else sel.split(",")
    out = f"{S}/grid_summary.md"
    for n in names:
        rc, log, bd, dt = run(n)
        row = summarise(n, rc, log, bd, dt)
        with open(out, "a") as f:
            f.write("| " + " | ".join(f"{k}={v}" for k, v in row.items()) + " |\n")
        print(json.dumps(row), flush=True)
