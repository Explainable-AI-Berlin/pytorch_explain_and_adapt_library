"""Status of a PEAL diffae generator run (default: the ImageNet CLIP ViT-L/14 DAE).

Usage: python utils/imagenet_dae_status.py [run_dir]
Prints step, throughput, loss / LPIPS curves, checkpoints, GPUs and cgroup memory.
"""

import glob, os, subprocess, sys, time
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.path.join(
        os.environ["PEAL_RUNS"], "imagenet/diffusion_autoencoder_openai_clip_vit_l14"
    )
)
print("run:", run, " now:", time.strftime("%Y-%m-%d %H:%M:%S"))
for stage in ("square64_ddim", "square64_autoenc_latent"):
    d = os.path.join(run, stage)
    files = sorted(glob.glob(d + "/events.out.tfevents.*"), key=os.path.getmtime)
    if not files:
        continue
    print(
        f"[{stage}] {len(files)} event file(s); newest {os.path.basename(files[-1])} mtime {time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(files[-1])))}"
    )
    ea = EventAccumulator(files[-1], size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags()["scalars"]
    if "loss" in tags:
        ev = ea.Scalars("loss")
        last = ev[-1]
        print(
            f"  loss: step {last.step}, last value {last.value:.4f}, mean of last 200 logged {sum(e.value for e in ev[-200:])/min(200,len(ev)):.4f}, n={len(ev)}"
        )
        for n in (50, 500):
            if len(ev) > n:
                a, b = ev[-n - 1], ev[-1]
                dt, ds = b.wall_time - a.wall_time, b.step - a.step
                if dt > 0:
                    print(
                        f"  rate over last {n} logs ({ds} steps, {dt/60:.1f} min): {ds/dt:.2f} it/s = {64*ds/dt:.0f} img/s"
                    )
    for t in (
        "rec_loss_enc/lpips",
        "rec_loss_enc_ema/lpips",
        "rec_loss/lpips",
        "rec_loss_ema/lpips",
        "lpips",
    ):
        if t in tags:
            ev = ea.Scalars(t)
            print(f"  {t}: " + ", ".join(f"{e.step}:{e.value:.3f}" for e in ev[-6:]))
    for p in sorted(glob.glob(d + "/*.ckpt")) + sorted(glob.glob(d + "/ema/*.ckpt")):
        print(
            f"  {os.path.relpath(p, run):40s} {os.path.getsize(p)/1e9:5.1f} GB  {time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(p)))}"
        )
    if os.path.exists(os.path.join(d, "latent.pkl")):
        print("  latent.pkl present")
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
    cg = "/sys/fs/cgroup/system.slice/slurmstepd.scope/s3PHM28DAH7C00/step_0/"
    print(
        "cgroup mem: %.1f GiB current, oom_kill=%s"
        % (
            int(open(cg + "memory.current").read()) / 2**30,
            [l for l in open(cg + "memory.events") if l.startswith("oom_kill")][
                0
            ].split()[1],
        )
    )
except Exception as e:
    print("cgroup:", e)
print(
    "du:",
    subprocess.run(["du", "-sh", run], capture_output=True, text=True).stdout.split()[
        0
    ],
)
