import os
import itertools

lr = 0.002
backward_t = 10
optimizer = "adam"

# Define the hyperparameters
dist_ps = ["none", "l1", "l2"]  # TODO: new scheme
dist_types = ["latent", "pixel"]
rmodel_paths = [
    "/home/tha/thesis_runs/regressor/imdb_wiki_densenet_linear_only-256/version_0/checkpoints/last.ckpt",
    "/home/tha/thesis_runs/regressor/imdb_wiki_densenet_fullft-256/version_0/checkpoints/last.ckpt",
]

# Iterate through all combinations of hyperparameters
for dist_p, dist_type, rmodel_path in itertools.product(
    dist_ps, dist_types, rmodel_paths
):
    command = f"sbatch scripts/run_diffeocf_dae_celebahq.sh {lr} {backward_t} {dist_p} {dist_type} {optimizer} {rmodel_path}"
    os.system(command)
