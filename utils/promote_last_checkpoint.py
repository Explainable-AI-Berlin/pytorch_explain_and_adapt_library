"""Promote the furthest-along valid last*.ckpt to last.ckpt before a resume.

Two Lightning behaviours combine badly here:

  * ModelCheckpoint will not clobber a last.ckpt it did not create in the
    current process, so every restart writes last-v1.ckpt, last-v2.ckpt, ...
  * diffae's train() unconditionally resumes from "<logdir>/last.ckpt".

Left alone, every restart rewinds to the checkpoint of the FIRST crash.

Rank by global_step, never by mtime: a restart that rewound and then saved
writes a NEWER file holding an OLDER step, so mtime ordering throws progress
away.
"""

import os
import sys
import zipfile
import glob


def valid(path):
    """A torch.save archive truncated by a SIGKILL fails to open as a zip."""
    try:
        with zipfile.ZipFile(path) as z:
            return any(n.endswith("data.pkl") for n in z.namelist())
    except Exception:
        return False


def step_of(path):
    """Read global_step without paging the 4.7 GB of tensor data into RAM."""
    import torch

    for kwargs in ({"mmap": True}, {}):
        try:
            d = torch.load(path, map_location="cpu", weights_only=False, **kwargs)
            step = d.get("global_step")
            del d
            return step
        except Exception:
            continue
    return None


def promote(logdir):
    last = os.path.join(logdir, "last.ckpt")
    cands = [p for p in glob.glob(os.path.join(logdir, "last*.ckpt")) if valid(p)]
    if not cands:
        print(f"[promote] {logdir}: no valid last*.ckpt found")
        return

    steps = {}
    for p in cands:
        s = step_of(p)
        if s is not None:
            steps[p] = s
        print(f"[promote]   candidate {os.path.basename(p)} global_step={s}")

    if not steps:
        print(
            f"[promote] {logdir}: could not read global_step from any candidate; "
            f"leaving last.ckpt untouched"
        )
        return

    best = max(steps, key=lambda p: steps[p])
    best_step = steps[best]

    if best != last:
        os.replace(best, last)
        print(
            f"[promote] {logdir}: {os.path.basename(best)} -> last.ckpt "
            f"(global_step={best_step})"
        )
    else:
        print(
            f"[promote] {logdir}: last.ckpt is already furthest along "
            f"(global_step={best_step})"
        )

    # Drop only copies we positively know are behind; each costs ~4.7 GB.
    for p, s in steps.items():
        if p in (best, last) or not os.path.exists(p):
            continue
        if s < best_step:
            try:
                os.remove(p)
                print(
                    f"[promote] removed {os.path.basename(p)} (step {s} < {best_step})"
                )
            except OSError as e:
                print(f"[promote] could not remove {p}: {e}")


def prune_step_ckpts(logdir, keep=3):
    """Keep only the `keep` furthest-along step=*.ckpt files.

    Lightning's save_top_k only prunes files the CURRENT process wrote, so the
    ones left behind by earlier runs -- including step=NNNN-v1.ckpt duplicates
    -- accumulate at ~4.7 GB each. Safe to do here: no trainer is running yet,
    and the next process starts with an empty top-k list.
    """
    import re

    found = []
    for f in glob.glob(os.path.join(logdir, "step=*.ckpt")):
        m = re.search(r"step=(\d+)", os.path.basename(f))
        if m:
            found.append((int(m.group(1)), os.path.getmtime(f), f))
    if len(found) <= keep:
        return
    found.sort(reverse=True)
    for _, _, f in found[keep:]:
        try:
            os.remove(f)
            print(f"[promote] pruned old {os.path.basename(f)}")
        except OSError as e:
            print(f"[promote] could not prune {f}: {e}")


if __name__ == "__main__":
    root = sys.argv[1]
    promote(root)
    prune_step_ckpts(root)
    ema = os.path.join(root, "ema")
    if os.path.isdir(ema):
        promote(ema)
        prune_step_ckpts(ema)
