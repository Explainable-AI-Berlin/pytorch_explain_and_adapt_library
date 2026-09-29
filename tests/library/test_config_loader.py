"""Placeholder expansion, nested-yaml inlining and the PEAL_BASE override."""

import os

import yaml

from peal.global_utils import (
    _load_yaml_config,
    get_project_resource_dir,
    load_yaml_config,
)


def test_peal_base_env_override(monkeypatch, tmp_path):
    monkeypatch.delenv("PEAL_BASE", raising=False)
    assert os.path.isdir(os.path.join(get_project_resource_dir(), "peal"))
    monkeypatch.setenv("PEAL_BASE", str(tmp_path))
    assert get_project_resource_dir() == str(tmp_path.resolve())


def test_placeholders_and_nested_yaml(monkeypatch, tmp_path):
    monkeypatch.setenv("PEAL_RUNS", str(tmp_path / "runs"))
    monkeypatch.setenv("PEAL_DATA", str(tmp_path / "data"))
    inner = tmp_path / "inner.yaml"
    inner.write_text(
        yaml.safe_dump(
            {"k": "$PEAL_DATA/x", "stage1_config": "$PEAL_RUNS/keep_as_path.yaml"}
        )
    )
    outer = tmp_path / "outer.yaml"
    outer.write_text(
        yaml.safe_dump({"child": str(inner), "runs": "$PEAL_RUNS/a", "plain": 3})
    )
    cfg = _load_yaml_config(str(outer))
    assert cfg["plain"] == 3
    assert cfg["runs"] == str(tmp_path / "runs" / "a")
    # a value ending in .yaml is inlined as a nested config ...
    assert isinstance(cfg["child"], dict) and cfg["child"]["k"] == str(
        tmp_path / "data" / "x"
    )
    # ... unless its key is in RAW_PATH_KEYS, where the path itself is the value
    assert cfg["child"]["stage1_config"] == str(tmp_path / "runs" / "keep_as_path.yaml")


def test_load_yaml_config_infers_the_model_and_falls_back_to_namespace():
    from peal.data.interfaces import DataConfig

    cfg = load_yaml_config({"config_name": "DataConfig", "input_size": [3, 32, 32]})
    assert isinstance(cfg, DataConfig) and cfg.input_size == [3, 32, 32]
    ns = load_yaml_config({"just": "a dict"})
    assert ns.just == "a dict"
