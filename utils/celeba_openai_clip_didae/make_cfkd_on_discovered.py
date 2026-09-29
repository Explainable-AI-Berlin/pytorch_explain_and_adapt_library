"""Write a CFKD config that finetunes on the directions a DiDAE run marked 'false'
(what DiDAE step 9 does, but as a separate process so two generators never share the
16 GiB cgroup).  Usage: make_cfkd_on_discovered.py <didae_run_dir> <out_yaml> <base_dir_name> [indices]
indices: optional comma list to override the run's false directions."""

import os, sys, re, yaml

BASE = os.environ["PEAL_BASE"]
run_dir, out_yaml, bd_name = sys.argv[1:4]
override = sys.argv[4] if len(sys.argv) > 4 else None
cfg = yaml.safe_load(
    open(
        f"{BASE}/configs/didae_experiments/adaptors/celeba1kx098_resnet18_procrustes_openai_clip_cfkd.yaml"
    )
)
didae = yaml.safe_load(open(f"{run_dir}/config.yaml"))
false_dirs = []
for line in open(f"{run_dir}/direction_feedback.txt"):
    m = re.match(r"direction=(\d+), feedback=(\w+)", line)
    if m and m.group(2) == "false":
        false_dirs.append(int(m.group(1)))
if override:
    false_dirs = [int(v) for v in override.split(",")]
sd = didae["sparse_dictionary"]
cfg["sparse_dictionary"] = (
    sd if isinstance(sd, str) else sd
)  # dict configs pass through
ex = didae["explainer"]
if isinstance(ex, str):
    cfg["explainer"] = ex
else:  # the run's config.yaml stores the expanded DAEdistillConfig; recover the sampler from it
    smp = (ex.get("sampler") or {}).get("type")
    scale = float(didae.get("component_bounds_scale", 1.0) or 1.0)
    suffix = f"_b{int(scale)}" if scale != 1.0 else ""
    cfg["explainer"] = (
        f"<PEAL_BASE>/configs/didae_experiments/explainers/dae_distill_ddpm{suffix}.yaml"
        if smp == "ddpm"
        else f"<PEAL_BASE>/configs/didae_experiments/explainers/dae_distill{suffix}.yaml"
    )
cfg["component_indices"] = false_dirs
cfg["base_dir"] = f"$PEAL_RUNS/celeba1k/Blond_Hair/classifier_poisoned098/{bd_name}"
yaml.safe_dump(cfg, open(out_yaml, "w"), sort_keys=False)
print(
    f"[cfkd-cfg] {out_yaml}: component_indices={false_dirs} explainer={cfg['explainer']} sd={str(cfg['sparse_dictionary'])[:80]}"
)
