import os
import itertools

# Define the hyperparameters
attack_methods = ["PGD"]
attack_steps = [1.0]
# From Paper: "For the age attribute, we used 0.15 for l1 and 0.05 for l2."
dist_l1s = [0.0, 0.15]
dist_l2s = [0.0, 0.05]
rmodel_paths = [
    "/home/tha/thesis_runs/regressor/imdb_wiki_densenet_linear_only-256/version_0/checkpoints/last.ckpt",
    "/home/tha/thesis_runs/regressor/imdb_wiki_densenet_fullft-256/version_0",
]

# Iterate through all combinations of hyperparameters
for attack_method, attack_step, dist_l1, dist_l2, rmodel_path in itertools.product(
    attack_methods, attack_steps, dist_l1s, dist_l2s, rmodel_paths
):
    command = f"sbatch scripts/regression/run_age_celebahq.sh {attack_method} {attack_step} {dist_l1} {dist_l2} {rmodel_path}"
    os.system(command)
