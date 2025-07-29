import os

for lr in [0.002, 0.005, 0.01]:
    cmd = f"bash scripts/run_diffeocf_dae_celebahq.sh {lr}"
    print(cmd)
    os.system(cmd)
