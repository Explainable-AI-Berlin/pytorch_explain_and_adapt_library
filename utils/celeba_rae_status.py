"""Status of the CelebA RAE generator run (RAEv2 two-stage, OpenAI CLIP ViT-L/14).

Usage: python utils/celeba_rae_status.py [run_dir]
       (default run_dir: $PEAL_RUNS/celeba/rae_clip)

Prints, per pipeline stage, what is finished, where the newest attempt stands
(tqdm position, step time, projected end), the newest loss line, the
checkpoints, plus the stage-2 latent cache, the GPUs and this allocation's
cgroup memory. RAEv2 writes no tensorboard events, so everything here is read
from the stage logs the pipeline appends to (stage1.log, stats.log, cache.log,
stage2.log) -- they hold every attempt, newest last.
"""

import glob
import os
import re
import subprocess
import sys
import time

import yaml

RUN = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.path.join(os.environ.get("PEAL_RUNS", "peal_runs"), "celeba/rae_clip")
)

TQDM = re.compile(
    r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([0-9:]+)<([0-9:?]+),\s*([0-9.]+)(s/it|it/s)"
)
LOSS = re.compile(r"(loss[\w/]*|lpips|psnr|ssim)\s*[:=]\s*(-?[0-9.]+)", re.I)


def tail_text(path, nbytes=400000):
    with open(path, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - nbytes))
        return f.read().decode("utf-8", "replace").replace("\r", "\n")


def fmt_time(seconds):
    return "%dh%02dm" % (seconds // 3600, (seconds % 3600) // 60)


def stage_report(name, log_name, exp_dir, epochs):
    log = os.path.join(RUN, log_name)
    print(f"\n[{name}]  epochs configured: {epochs}")
    ckpts = sorted(glob.glob(os.path.join(exp_dir, "checkpoints", "ep-*.pt")))
    if epochs is not None:
        final = os.path.join(exp_dir, "checkpoints", f"ep-{int(epochs):07d}.pt")
        print("  finished:", os.path.isfile(final))
    for p in ckpts[-4:]:
        print(
            f"    {os.path.basename(p):20s} {os.path.getsize(p) / 1e9:5.1f} GB  "
            f"{time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(p)))}"
        )
    if not os.path.isfile(log):
        print("  no log yet:", log)
        return
    text = tail_text(log)
    attempts = [l for l in text.splitlines() if l.startswith("===== ")]
    if attempts:
        print("  newest attempt started:", attempts[-1][6:25])
    bars = TQDM.findall(text)
    if bars:
        pct, done, total, elapsed, eta, rate, unit = bars[-1]
        s_it = float(rate) if unit == "s/it" else 1.0 / max(float(rate), 1e-9)
        left = (int(total) - int(done)) * s_it
        print(
            f"  step {done}/{total} ({pct}%), {s_it:.2f} s/step, elapsed {elapsed}, "
            f"tqdm eta {eta}, ends in {fmt_time(left)} (~{time.strftime('%m-%d %H:%M', time.localtime(time.time() + left))})"
        )
    # RAEv2 logs the losses as the tqdm postfix ("..., 3.01s/it, loss=1.46, lr=...")
    postfix = [
        l.split("it,", 1)[1].rstrip("] ")
        for l in text.splitlines()
        if l.startswith("Training:") and "it," in l and LOSS.search(l)
    ]
    if postfix:
        print("  losses:" + postfix[-1][:200])
    for l in [l for l in text.splitlines() if LOSS.search(l) and "%|" not in l][-2:]:
        print("  " + l.strip()[:200])
    bad = [
        l
        for l in text.splitlines()[-4000:]
        if re.search(r"Traceback|CUDA out of memory|Killed|Error:", l)
    ]
    for l in bad[-3:]:
        print("  !! " + l.strip()[:200])


def main():
    print("run:", RUN, " now:", time.strftime("%Y-%m-%d %H:%M:%S"))
    cfg_path = os.path.join(RUN, "config.yaml")
    if not os.path.isfile(cfg_path):
        cfg_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs/didae_experiments/generators/celeba_rae_clip.yaml",
        )
    cfg = yaml.safe_load(open(cfg_path))
    peal_base = os.environ.get(
        "PEAL_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    def stage_yaml(key):
        p = str(cfg[key]).replace("<PEAL_BASE>", peal_base)
        return yaml.safe_load(open(p))

    s1, s2 = stage_yaml("stage1_config"), stage_yaml("stage2_config")
    stage_report(
        "stage 1  RAE decoder",
        "stage1.log",
        os.path.join(RUN, "stage1", cfg["stage1_experiment"]),
        s1["training"]["epochs"],
    )

    assets = os.path.join(RUN, "stage1_assets")
    print("\n[stats]  stage-1 assets")
    for f in ("decoder.pt", "stats.pt"):
        p = os.path.join(assets, f)
        print(
            f"  {f:12s} {'%.1f GB  %s' % (os.path.getsize(p) / 1e9, time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(p)))) if os.path.isfile(p) else 'MISSING'}"
        )

    cache = cfg["cache_dir"]
    meta = os.path.join(cache, "metadata.json")
    shards = len(glob.glob(os.path.join(cache, "**", "*.npz"), recursive=True)) or len(
        glob.glob(os.path.join(cache, "**", "*.pt"), recursive=True)
    )
    du = subprocess.run(
        ["du", "-sh", cache], capture_output=True, text=True
    ).stdout.split()
    print(
        f"\n[cache]  {cache}: complete={os.path.isfile(meta)}, {shards} shard files, {du[0] if du else '-'}"
    )

    stage_report(
        "stage 2  DDT flow matching",
        "stage2.log",
        os.path.join(RUN, "stage2", cfg["stage2_experiment"]),
        s2["training"]["epochs"],
    )

    print()
    print(
        subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    try:
        cg = "/sys/fs/cgroup" + open("/proc/self/cgroup").read().strip().split(":")[-1]
        cg = cg.rsplit("/step_", 1)[0] + "/step_0/user"
        oom = [l for l in open(cg + "/memory.events") if l.startswith("oom_kill")][
            0
        ].split()[1]
        print(
            "cgroup mem: %.1f of %.1f GiB, oom_kill=%s"
            % (
                int(open(cg + "/memory.current").read()) / 2**30,
                int(open(cg + "/memory.max").read()) / 2**30,
                oom,
            )
        )
    except Exception as e:
        print("cgroup:", e)
    for p in sorted(glob.glob(os.path.join(peal_base, "logs", "celeba_rae_clip*.out"))):
        print(
            "driver log:",
            p,
            "->",
            (
                tail_text(p, 4000).strip().splitlines()[-1][:160]
                if os.path.getsize(p)
                else "(empty)"
            ),
        )


if __name__ == "__main__":
    main()
