"""What preflight must keep catching.

Every case here is a failure that reached a real run before the check existed,
so a refactor that drops one of them is a regression with a known cost.
"""

import os

import yaml

from peal.preflight import (
    check_config,
    check_script_executables,
    configs_in_script,
    scripts_invoked_by,
)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))
    return str(path)


def test_missing_input_is_a_problem(monkeypatch, tmp_path):
    monkeypatch.setenv("PEAL_RUNS", str(tmp_path / "runs"))
    cfg = write(tmp_path / "a.yaml", {"student": "$PEAL_RUNS/nowhere/model.cpl"})
    problems, _ = check_config(cfg)
    assert any("student does not exist" in p for p in problems)


def test_nested_block_path_is_checked(monkeypatch, tmp_path):
    """`teacher:` is a scalar on some datasets and an inline block on others.

    The block form used to be skipped whole, so a missing teacher checkpoint only
    surfaced at runtime -- which is how two tabular configs shipped unrunnable.
    """
    monkeypatch.setenv("PEAL_RUNS", str(tmp_path / "runs"))
    cfg = write(
        tmp_path / "a.yaml",
        {
            "teacher": {
                "type": "symbolic",
                "model": "$PEAL_RUNS/nowhere/model.cpl",
                "confounder_name": "sex",
            }
        },
    )
    problems, _ = check_config(cfg)
    assert any("teacher.model does not exist" in p for p in problems)


def test_scalar_teacher_without_a_slash_is_not_a_path(tmp_path):
    """`Baseline:false`, `human@8000`, `SegmentationMask` are teacher names."""
    cfg = write(tmp_path / "a.yaml", {"teacher": "Baseline:false"})
    problems, notes = check_config(cfg)
    assert problems == []
    assert not any("teacher" in n for n in notes)


def test_relative_input_path_is_noted(monkeypatch, tmp_path):
    """It resolves against the cwd, so it works only by accident."""
    monkeypatch.setenv("PEAL_RUNS", str(tmp_path / "runs"))
    present = tmp_path / "peal_runs" / "model.cpl"
    present.parent.mkdir(parents=True)
    present.write_text("x")
    monkeypatch.chdir(tmp_path)
    cfg = write(tmp_path / "a.yaml", {"student": "peal_runs/model.cpl"})
    problems, notes = check_config(cfg)
    assert problems == []
    assert any("relative path" in n and "cwd" in n for n in notes)


def test_dataset_path_is_read_out_of_the_data_config(monkeypatch, tmp_path):
    """A data config that exists can still name a dataset that does not.

    Notes rather than problems: the tabular dataset classes download what is
    missing, so an absent dataset does not stop a run -- it only needs network
    access the compute node may not have.
    """
    monkeypatch.setenv("PEAL_DATA", str(tmp_path / "data"))
    data_cfg = write(tmp_path / "data.yaml", {"dataset_path": "$PEAL_DATA/absent"})
    cfg = write(tmp_path / "a.yaml", {"data": data_cfg})
    problems, notes = check_config(cfg)
    assert problems == []
    assert any("dataset is not on this machine" in n for n in notes)


def test_relative_dataset_path_is_noted(monkeypatch, tmp_path):
    """With a relative dataset_path a downloading class writes into the repo."""
    monkeypatch.setenv("PEAL_DATA", str(tmp_path / "data"))
    data_cfg = write(tmp_path / "data.yaml", {"dataset_path": "datasets/adult"})
    cfg = write(tmp_path / "a.yaml", {"data": data_cfg})
    problems, notes = check_config(cfg)
    assert problems == []
    assert any("resolves against" in n and "working directory" in n for n in notes)


def test_conflict_markers_are_a_problem(tmp_path):
    path = tmp_path / "a.yaml"
    path.write_text("student: x\n<<<<<<< HEAD\nfoo: 1\n")
    problems, _ = check_config(str(path))
    assert any("merge-conflict" in p for p in problems)


def test_input_produced_upstream_is_not_a_problem(monkeypatch, tmp_path):
    """The scripts are ordered: they train a generator, then run what consumes it.

    Without this an unrun script looks like a broken one -- it reported 40
    configs broken when the real number was 7.
    """
    monkeypatch.setenv("PEAL_RUNS", str(tmp_path / "runs"))
    target = str(tmp_path / "runs" / "gen" / "config.yaml")
    cfg = write(tmp_path / "a.yaml", {"generator": "$PEAL_RUNS/gen/config.yaml"})
    problems, notes = check_config(cfg, will_exist={target})
    assert problems == []
    assert any("produced by an earlier line" in n for n in notes)


def test_foreign_path_is_a_problem(tmp_path):
    cfg = write(
        tmp_path / "a.yaml", {"student": "/home/space/datasets/peal_ahmed/m.cpl"}
    )
    problems, _ = check_config(cfg)
    assert any("personal path" in p for p in problems)


def test_configs_in_script_expands_for_loops(tmp_path):
    """`${METHOD}` is not a missing config; it is three configs."""
    script = tmp_path / "s.sh"
    script.write_text(
        "for METHOD in ace dime sce; do\n"
        '  python run_cfkd.py --config "configs/${METHOD}_cfkd.yaml"\n'
        "done\n"
        "# python run_cfkd.py --config configs/commented_out.yaml\n"
    )
    found = configs_in_script(str(script))
    assert found == [
        "configs/ace_cfkd.yaml",
        "configs/dime_cfkd.yaml",
        "configs/sce_cfkd.yaml",
    ]


def test_scripts_invoked_by_finds_targets_and_skips_modules(tmp_path):
    script = tmp_path / "s.sh"
    script.write_text(
        "python train_predictor.py --config a.yaml\n"
        "python -u run_cfkd.py --config b.yaml\n"
        "python3 tools/helper.py\n"
        "python -m peal.entrypoints.run_cfkd --config c.yaml\n"
        "# python never_run.py\n"
    )
    found = scripts_invoked_by(str(script))
    assert found == ["train_predictor.py", "run_cfkd.py", "tools/helper.py"]
    assert "never_run.py" not in found


def test_check_script_executables_reports_a_missing_target(tmp_path):
    """Three files the scripts invoke had never been committed; a clone died on them."""
    script = tmp_path / "s.sh"
    script.write_text("python definitely_absent_%s.py\n" % os.getpid())
    problems, _ = check_script_executables(str(script))
    assert any("does not exist" in p for p in problems)


def test_generator_with_a_config_but_no_weights_is_a_problem(monkeypatch, tmp_path):
    """The retention sweep of 2026-09-16 emptied most DDPM directories.

    `config.yaml` survived every one of them, so a config-exists check passed a
    generator that cannot generate and the run only failed once it tried to load.
    """
    monkeypatch.setenv("PEAL_RUNS", str(tmp_path / "runs"))
    gen = tmp_path / "runs" / "ddpm"
    gen.mkdir(parents=True)
    (gen / "config.yaml").write_text("is_trained: true\n")
    cfg = write(tmp_path / "a.yaml", {"generator": "$PEAL_RUNS/ddpm/config.yaml"})
    problems, _ = check_config(cfg)
    assert any("config but no weights" in p for p in problems)


def test_npz_counts_as_weights(monkeypatch, tmp_path):
    """A sparse dictionary stores itself as `weights.npz`, not a torch checkpoint.

    Leaving .npz out of the suffix list made the check above report every Procrustes
    and SVD dictionary in the tree as weightless -- 13 false positives in one run.
    """
    monkeypatch.setenv("PEAL_RUNS", str(tmp_path / "runs"))
    d = tmp_path / "runs" / "OrthogonalProcrustesDictionary"
    d.mkdir(parents=True)
    (d / "config.yaml").write_text("k: 1\n")
    (d / "weights.npz").write_bytes(b"\x00")
    cfg = write(
        tmp_path / "a.yaml",
        {"sparse_dictionary": "$PEAL_RUNS/OrthogonalProcrustesDictionary/config.yaml"},
    )
    problems, _ = check_config(cfg)
    assert not any("no weights" in p for p in problems)


def test_weights_one_level_down_count(monkeypatch, tmp_path):
    """Generators keep their weights in `ema/` or `model/`, not beside the config."""
    monkeypatch.setenv("PEAL_RUNS", str(tmp_path / "runs"))
    gen = tmp_path / "runs" / "ddpm"
    (gen / "ema").mkdir(parents=True)
    (gen / "config.yaml").write_text("is_trained: true\n")
    (gen / "ema" / "0100.pt").write_bytes(b"\x00")
    cfg = write(tmp_path / "a.yaml", {"generator": "$PEAL_RUNS/ddpm/config.yaml"})
    problems, _ = check_config(cfg)
    assert not any("no weights" in p for p in problems)
