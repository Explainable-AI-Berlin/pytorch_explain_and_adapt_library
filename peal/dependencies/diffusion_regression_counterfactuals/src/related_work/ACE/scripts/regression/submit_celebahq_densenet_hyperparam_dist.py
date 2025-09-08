import os
import itertools

# Define the hyperparameters
attack_method = "PGD"
attack_step = 1.0
# From Paper: "For the age attribute, we used 0.15 for l1 and 0.05 for l2."
dist_l1s = [0, 0.01, 0.001, 0.0001, 0.00001, 0.15]
dist_l2s = [0.01, 0.001, 0.0001, 0.00001, 0.05]
rmodel_path = "/home/tha/thesis_runs/regressor/imdb_wiki_densenet_linear_only-256/version_0/checkpoints/last.ckpt"


for dist_l1, dist_l2 in [(dl1, 0) for dl1 in dist_l1s] + [(0, dl2) for dl2 in dist_l2s]:
    command = f"sbatch scripts/regression/run_age_celebahq.sh {attack_method} {attack_step} {dist_l1} {dist_l2} {rmodel_path}"
    os.system(command)
