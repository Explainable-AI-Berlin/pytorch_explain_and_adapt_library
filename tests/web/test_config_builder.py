import pytest
import yaml

from peal.web.config_builder import build_job_configs, resolve_normalization


def test_normalization_presets():
    assert resolve_normalization({"normalization": "zero_one"}) is None
    assert resolve_normalization({"normalization": "minus_one_one"}) == [0.5, 0.5]
    assert resolve_normalization(
        {"normalization": "custom", "mean": [0, 0, 0], "std": [1, 1, 1]}
    ) == [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    with pytest.raises(ValueError):
        resolve_normalization(
            {"normalization": "custom", "mean": [0, 0], "std": [1, 1]}
        )
    with pytest.raises(ValueError):
        resolve_normalization({"normalization": "nope"})


def test_build_job_configs(tmp_path, monkeypatch):
    job = tmp_path / "job"
    (job / "dataset").mkdir(parents=True)
    (job / "dataset" / "data.csv").write_text("ImgPath,Class\n0/a.png,0\n1/b.png,1\n")
    (job / "model.onnx").write_bytes(b"x")
    fields = {
        "input_height": 224,
        "input_width": 224,
        "normalization": "imagenet",
        "class_a_name": "cheeseburger",
        "class_b_name": "hotdog",
    }
    monkeypatch.delenv("PEAL_RAE_WEIGHTS", raising=False)
    # Without $PEAL_RAE_WEIGHTS the template's published weights are used.
    default = yaml.safe_load(open(build_job_configs(str(job), fields, n_samples=2)))
    assert yaml.safe_load(open(default["generator"]))["weights"].startswith("hf://")
    assert default["max_decode_directions"] == 10 == default["top_k_directions"]
    path = build_job_configs(
        str(job), fields, n_samples=2, rae_weights="hf://org/peal-rae"
    )
    cfg = yaml.safe_load(open(path))
    assert cfg["service_mode"] is True
    assert cfg["student"] == str(job / "model.onnx")
    assert cfg["teacher"]["type"] == "web" and cfg["teacher"]["dir"] == str(
        job / "feedback"
    )
    assert cfg["cfkd_teacher"] == "Baseline:false"
    assert cfg["export_onnx"] is True
    # single atoms, minimal edits (bounds as observed, each edit shortened to
    # 2x the boundary step, random choice of rendered flips), every flip of the
    # rendered directions
    assert cfg["concept_replacement"] is False
    assert cfg["component_bounds_scale"] == 1.0
    assert cfg["edit_depth_factor"] == 2.0
    assert cfg["decode_selection"] == "random"
    assert cfg["max_decode_per_direction"] is None
    assert cfg["base_dir"] == str(job / "run")
    data = yaml.safe_load(open(cfg["data"]))
    assert data["input_size"] == [3, 224, 224]
    assert data["normalization"] == [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    assert data["dataset_path"] == str(job / "dataset")
    gen = yaml.safe_load(open(cfg["generator"]))
    assert gen["weights"] == "hf://org/peal-rae"
    assert gen["data"]["dataset_path"] == str(job / "dataset")
    assert gen["generator_type"] == "RAEDiffusionAutoencoder"


def test_job_configs_load_through_peal_models(tmp_path):
    """The three yamls the builder writes must validate against PEAL's config
    models (DiDAEConfig, the RAE generator config, DataConfig) without running
    anything: this is what run_didae.py does first."""
    pytest.importorskip("peal.adaptors.didae")
    from peal.adaptors.didae import DiDAEConfig
    from peal.global_utils import load_yaml_config

    job = tmp_path / "job"
    (job / "dataset").mkdir(parents=True)
    (job / "dataset" / "data.csv").write_text("ImgPath,Class\n0/a.png,0\n1/b.png,1\n")
    (job / "model.onnx").write_bytes(b"x")
    fields = {
        "input_height": 224,
        "input_width": 224,
        "normalization": "zero_one",
        "class_a_name": "a",
        "class_b_name": "b",
    }
    path = build_job_configs(
        str(job), fields, n_samples=2, rae_weights="hf://org/peal-rae"
    )
    cfg = load_yaml_config(path, DiDAEConfig)
    assert cfg.service_mode is True and cfg.max_decode_per_direction is None
    assert cfg.generator.weights == "hf://org/peal-rae"
    assert cfg.generator.generator_type == "RAEDiffusionAutoencoder"
    assert cfg.data.dataset_path == str(job / "dataset")
    assert cfg.data.normalization is None
    assert cfg.teacher["type"] == "web"
    assert cfg.explainer.explainer_type == "DAEdistill"
    assert cfg.task.output_channels == 2


def test_render_directions_field(tmp_path):
    job = tmp_path / "job"
    (job / "dataset").mkdir(parents=True)
    (job / "dataset" / "data.csv").write_text("ImgPath,Class\n0/a.png,0\n1/b.png,1\n")
    (job / "model.onnx").write_bytes(b"x")
    fields = {
        "input_height": 224,
        "input_width": 224,
        "normalization": "imagenet",
        "render_directions": 3,
    }
    cfg = yaml.safe_load(
        open(build_job_configs(str(job), fields, n_samples=2, rae_weights="hf://o/r"))
    )
    assert cfg["max_decode_directions"] == 3 and cfg["top_k_directions"] == 3


def test_render_per_direction_field(tmp_path):
    job = tmp_path / "job"
    (job / "dataset").mkdir(parents=True)
    (job / "dataset" / "data.csv").write_text("ImgPath,Class\n0/a.png,0\n1/b.png,1\n")
    (job / "model.onnx").write_bytes(b"x")
    fields = {"input_height": 224, "input_width": 224, "normalization": "imagenet"}
    cfg = yaml.safe_load(
        open(build_job_configs(str(job), fields, n_samples=2, rae_weights="hf://o/r"))
    )
    assert cfg["max_decode_per_direction"] is None  # 0 / missing: every flip
    assert cfg["max_export_per_direction"] is None
    fields["render_per_direction"] = 30
    cfg = yaml.safe_load(
        open(build_job_configs(str(job), fields, n_samples=2, rae_weights="hf://o/r"))
    )
    assert cfg["max_decode_per_direction"] == 30


def test_sweep_uses_generator_steps_and_pairs(tmp_path):
    job = tmp_path / "job"
    (job / "dataset").mkdir(parents=True)
    (job / "dataset" / "data.csv").write_text("ImgPath,Class\n0/a.png,0\n1/b.png,1\n")
    (job / "model.onnx").write_bytes(b"x")
    fields = {"input_height": 224, "input_width": 224, "normalization": "imagenet"}
    cfg = yaml.safe_load(
        open(build_job_configs(str(job), fields, n_samples=2, rae_weights="hf://o/r"))
    )
    gen = yaml.safe_load(open(cfg["generator"]))
    # the sweep's set_sampler(explainer.sampler) must see the generator's steps
    assert cfg["explainer"]["sampler"] == gen["sampler"]
    assert "<PEAL_BASE>" not in str(cfg["explainer"])
    assert cfg["concept_replacement"] is False
    fields["direction_type"] = "pairs"
    cfg = yaml.safe_load(
        open(build_job_configs(str(job), fields, n_samples=2, rae_weights="hf://o/r"))
    )
    assert cfg["concept_replacement"] is True


def test_correction_field(tmp_path):
    job = tmp_path / "job"
    (job / "dataset").mkdir(parents=True)
    (job / "dataset" / "data.csv").write_text("ImgPath,Class\n0/a.png,0\n1/b.png,1\n")
    (job / "model.onnx").write_bytes(b"x")
    fields = {"input_height": 224, "input_width": 224, "normalization": "imagenet"}
    for correction, iterations in (("none", 0), ("dfr", 0), ("finetune", 1)):
        fields["correction"] = correction
        cfg = yaml.safe_load(
            open(
                build_job_configs(str(job), fields, n_samples=2, rae_weights="hf://o/r")
            )
        )
        assert cfg["finetune_iterations"] == iterations
        assert cfg["run_cfkd_on_false_directions"] is bool(iterations)
