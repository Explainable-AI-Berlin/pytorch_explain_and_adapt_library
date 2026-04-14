import os


GMODEL_PATH = "/home/tha/diffae/checkpoints/ffhq256_autoenc/last.ckpt"
RMODEL_PATH = "/home/tha/thesis_runs/regressor/imdb_wiki_densenet_linear_only-256/version_0/checkpoints/last.ckpt"
RORACLE_PATH = "/home/tha/thesis_runs/regressor/imdb_wiki_densenet_fullft-256/version_0/checkpoints/last.ckpt"


def construct_args(dist, dist_type):
    args = {
        "gmodel_path": GMODEL_PATH,
        "rmodel_path": RMODEL_PATH,
        "roracle_path": RORACLE_PATH,
        "confidence_threshold": 0.05,
        "dist": dist,
        "dist_type": dist_type,
        "optimizer": "adam",
        "forward_t": 250,
        "backward_t": 10,
        "num_steps": 200,
        "lr": 0.002,
        "image_folder": "/data/CelebAMask-HQ/",
        "limit_samples": 32,
        "batch_size": 4,
    }

    name = (
        f"CelebaHQ_FR-lr={args['lr']}-bt={args['backward_t']}-dist={args['dist']}+"
        f"{args['dist_type']}-opt={args['optimizer']}"
    )
    args["result_dir"] = f"/home/tha/thesis_runs/cf/celebahq/dists/{name}"

    print("Running with args:", args)
    return args


def get_full_sbatch_cmd(args: dict):
    dist_name = f"{args['dist']}+{args['dist_type']}"
    sbatch_script_header = f"""#!/bin/bash
#SBATCH --job-name=dcf_celebahq_dist={dist_name}
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --constraint=80gb
#SBATCH --output=logs/job-%j-{dist_name}.out
#SBATCH --chdir=/home/tha/master-thesis-xai/diff_cf_ir/scripts/cf/celebahq
#SBATCH --signal=SIGUSR1@600

"""

    args_string = " ".join([f"--{key}={value}" for key, value in args.items()])
    cmd = " ".join(
        [
            "apptainer run",
            "-B /home/space/datasets-sqfs/CelebAMask-HQ.sqfs:/data/CelebAMask-HQ:image-src=/",
            "--nv",
            "~/apptainers/thesis.sif",
            "python run_diffeocf_dae_celebahq.py",
            args_string,
        ]
    )

    print("Submitting", cmd)
    sbatch_cmd = sbatch_script_header + cmd
    # Use a here document to pass the whole pseudo script to sbatch
    return f"sbatch <<'EOF'\n{sbatch_cmd}\nEOF"


test_dists_latent = [
    ("none", "latent"),
    ("l1=0.0001", "latent"),
    ("l1=0.001", "latent"),
    ("l1=0.015", "latent"),  # Like ACE/10, it has some effect
    ("l2=0.0001", "latent"),
    ("l2=0.001", "latent"),
    ("l2=0.005", "latent"),  # Like ACE/10, but a bit too strong
]
test_dists_pixel = [
    ("l1=0.00001", "pixel"),
    ("l1=0.0001", "pixel"),
    ("l1=0.001", "pixel"),
    ("l1=0.15", "pixel"),  # Like ACE, too strong. No changes to the image at all
    ("l2=0.00001", "pixel"),
    ("l2=0.0001", "pixel"),
    ("l2=0.001", "pixel"),
    ("l2=0.005", "pixel"),  # Like ACE, too strong. No changes to the image at all
]

for dist, dist_type in test_dists_latent + test_dists_pixel:
    args = construct_args(dist, dist_type)
    sbatch_cmd = get_full_sbatch_cmd(args)
    os.system(sbatch_cmd)
