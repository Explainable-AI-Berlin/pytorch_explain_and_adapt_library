"""
Turn a web job (uploaded ONNX + ingested two-class dataset + the mandatory
form fields) into the three yaml files run_didae.py needs:

    <job>/data.yaml        Image2MixedDataset over <job>/dataset
    <job>/generator.yaml   the RAE generator with the published weights
    <job>/config.yaml      DiDAE, service mode, web feedback teacher

The templates live in configs/web_demo; only job-specific keys are filled in
here so the budget can be tuned in the template.
"""

import copy
import os

import yaml

from peal.web.cache import cache_path
from peal.web.paths import get_project_resource_dir

NORMALIZATION_PRESETS = {
    # value of DataConfig.normalization ([mean, std]; None = the model takes [0, 1])
    "imagenet": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    "zero_one": None,
    "minus_one_one": [0.5, 0.5],
    "clip": [[0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]],
}


def _templates_dir():
    return os.path.join(get_project_resource_dir(), "configs", "web_demo")


def _load_template(name):
    with open(os.path.join(_templates_dir(), name)) as f:
        return yaml.safe_load(f)


def resolve_normalization(fields):
    """``fields["normalization"]`` is a preset name or "custom" with
    ``fields["mean"]`` / ``fields["std"]`` (3 floats each)."""
    preset = fields.get("normalization")
    if preset == "custom":
        mean = [float(v) for v in fields["mean"]]
        std = [float(v) for v in fields["std"]]
        if len(mean) != 3 or len(std) != 3 or any(s <= 0 for s in std):
            raise ValueError("custom normalization needs 3 means and 3 positive stds")
        return [mean, std]
    if preset not in NORMALIZATION_PRESETS:
        raise ValueError(
            f"unknown normalization {preset!r}; one of {sorted(NORMALIZATION_PRESETS)} or custom"
        )
    return copy.deepcopy(NORMALIZATION_PRESETS[preset])


def build_job_configs(job_dir, fields, n_samples, rae_weights=None):
    """Write data.yaml, generator.yaml and config.yaml into ``job_dir``.

    ``fields`` (validated by the app): input_height, input_width, normalization
    (+ mean/std), class_a_name, class_b_name. ``n_samples`` is the size of the
    ingested two-class dataset. ``rae_weights`` overrides $PEAL_RAE_WEIGHTS.
    Returns the path of config.yaml.
    """
    job_dir = os.path.abspath(job_dir)
    dataset_dir = os.path.join(job_dir, "dataset")
    model_path = os.path.join(job_dir, "model.onnx")
    if not os.path.isfile(os.path.join(dataset_dir, "data.csv")):
        raise FileNotFoundError(
            f"{dataset_dir}/data.csv missing; ingest the dataset first"
        )
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"{model_path} missing")

    h, w = int(fields["input_height"]), int(fields["input_width"])
    data_cfg = {
        "dataset_class": "Image2MixedDataset",
        "dataset_path": dataset_dir,
        "num_samples": int(n_samples),
        "input_type": "image",
        "input_size": [3, h, w],
        "output_type": "singleclass",
        "output_size": [2],
        "downsize": "Resize",
        "x_selection": "imgs",
        "confounding_factors": ["Class"],
        "normalization": resolve_normalization(fields),
    }
    # The two class names the form collects are NOT written here. DataConfig has no
    # class_names field and ignores extra keys, so the line that used to set it was
    # silently dropped and gave the false impression that collages would be labelled
    # with them. Collage labels come from the dataset's CSV header (see
    # peal/data/datasets.py, self.attributes), which ingest writes as "ImgPath,Class".
    # The names still reach the UI and the judging prompts through job["fields"].
    data_path = os.path.join(job_dir, "data.yaml")
    with open(data_path, "w") as f:
        yaml.safe_dump(data_cfg, f, sort_keys=False)

    gen_cfg = _load_template("imagenet_rae_clip_hf.yaml")
    weights = (
        rae_weights or os.environ.get("PEAL_RAE_WEIGHTS") or gen_cfg.get("weights")
    )
    if not weights or "CHANGE_ME" in str(weights):
        raise ValueError(
            "no RAE weights configured: set $PEAL_RAE_WEIGHTS to hf://<org>/<repo> "
            "or a folder written by tools/export_rae_weights.py"
        )
    gen_cfg["weights"] = weights
    gen_cfg["data"]["dataset_path"] = dataset_dir
    gen_cfg["data"]["num_samples"] = int(n_samples)
    gen_cfg["base_path"] = os.path.join(job_dir, "generator_state")
    gen_path = os.path.join(job_dir, "generator.yaml")
    with open(gen_path, "w") as f:
        yaml.safe_dump(gen_cfg, f, sort_keys=False)

    cfg = _load_template("didae_service_template.yaml")
    cfg["student"] = model_path
    # written by the accuracy check (peal.web.cache); step 1 falls back to the
    # student when it is missing
    cfg["student_logits_cache"] = cache_path(job_dir)
    cfg["teacher"] = {
        "type": "web",
        "dir": os.path.join(job_dir, "feedback"),
        "timeout_s": float(
            os.environ.get("PEAL_WEB_FEEDBACK_TIMEOUT_S", str(48 * 3600))
        ),
    }
    cfg["data"] = data_path
    cfg["test_data"] = data_path
    cfg["generator"] = gen_path
    cfg["base_dir"] = os.path.join(job_dir, "run")
    render_directions = int(
        fields.get("render_directions") or cfg["max_decode_directions"]
    )
    cfg["max_decode_directions"] = render_directions
    cfg["top_k_directions"] = render_directions
    # 0 keeps the template's value (null: every latent flip)
    if int(fields.get("render_per_direction") or 0) > 0:
        cfg["max_decode_per_direction"] = int(fields["render_per_direction"])
    # step 9: full CFKD finetuning only when asked for; DFR runs after the
    # DiDAE process (peal/web/dfr.py), no correction stops after the verdicts
    full = fields.get("correction") == "finetune"
    cfg["finetune_iterations"] = 1 if full else 0
    # with 0 iterations DiDAE still builds the CFKD counterfactual dataset
    # (~1 h on Waterbirds, 2026-09-29), so switch step 9 off entirely
    cfg["run_cfkd_on_false_directions"] = full
    if fields.get("direction_type") == "pairs":
        # concept replacement: switch one concept off and another on together
        cfg["concept_replacement"] = True
        cfg["concept_replacement_unique"] = True
        cfg["concept_replacement_candidates"] = 64
    # The sweep calls generator.set_sampler(explainer.sampler) and an explainer
    # without a sampler resets the step count to 50, whatever generator.yaml
    # says; hand the generator's sampler to the explainer so it is used.
    explainer_path = str(cfg["explainer"]).replace(
        "<PEAL_BASE>", get_project_resource_dir()
    )
    with open(explainer_path) as f:
        explainer = yaml.safe_load(f)
    for key, value in list(explainer.items()):
        if isinstance(value, str) and "<PEAL_BASE>" in value:
            explainer[key] = value.replace("<PEAL_BASE>", get_project_resource_dir())
    explainer["sampler"] = dict(gen_cfg["sampler"])
    cfg["explainer"] = explainer
    cfg.setdefault("task", {})["y_selection"] = ["Class"]
    cfg["task"]["output_channels"] = 2
    config_path = os.path.join(job_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return config_path
