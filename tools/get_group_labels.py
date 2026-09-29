"""Label a dataset's samples as confounded or clean with SpRAy.

Drives the ``Spray`` teacher in ``peal/teachers/spray_teacher.py``: it
computes LRP attributions of the configured predictor at the configured
layer, spectrally clusters them, opens a ViRelAy project on the clusters and
turns the clusters selected there into per-sample group labels.

Invocation::

    python tools/get_group_labels.py --config <spray.yaml>

The YAML is a ``SprayConfig`` (``base_dir``, ``data``, ``model``, ``task``,
``classes_total``, ``attribution_layer``, ...). Everything is written into
``<base_dir>/attrbs-layer-<layer>_analysis/``: ``config.yaml``, the
``input.h5`` / ``attribution.h5`` / ``heatmaps.h5`` / ``concept_importance.h5``
databases, the ViRelAy project, ``data_spray_labels.csv`` (the dataset's
``data.csv`` with an added ``SprayLabel`` column, -1 for unlabelled rows) and
``result_summary.txt`` with the agreement between the SpRAy labels and the
dataset's ``has_confounder`` flag.
"""

import argparse

import os
import sys

# Runnable from a clone without installing PEAL: put the repository root on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from peal.global_utils import load_yaml_config
from peal.teachers.spray_teacher import SprayConfig, Spray


def main():
    """Load the ``SprayConfig`` and run the SpRAy teacher once.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    spray_config = load_yaml_config(args.config, config_model=SprayConfig)
    spray_teacher = Spray(spray_config)
    spray_teacher.get_feedback()


if __name__ == "__main__":
    main()
