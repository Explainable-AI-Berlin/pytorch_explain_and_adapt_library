import importlib.util
import os

import pytest
import torch


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "export_rae_weights",
        os.path.join(
            os.path.dirname(__file__), "..", "..", "tools", "export_rae_weights.py"
        ),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_extract_stage2_ema(tmp_path):
    tool = _load_tool()
    ema = {
        "module.a.weight": torch.ones(2, 2),
        "module.b": torch.zeros(3),
        "steps": torch.tensor(5),
    }
    ckpt = {
        "model": {"module.a.weight": torch.zeros(2, 2)},
        "ema": ema,
        "opt": {"x": 1},
        "epoch": 43,
        "step": 1000,
    }
    torch.save(ckpt, tmp_path / "ep.pt")
    n, meta = tool.extract_stage2_ema(
        str(tmp_path / "ep.pt"), str(tmp_path / "out.pt"), "ema", "bf16"
    )
    assert n == 3 and meta == {"epoch": 43, "step": 1000}
    sd = torch.load(tmp_path / "out.pt")
    assert set(sd) == {"a.weight", "b", "steps"}
    assert sd["a.weight"].dtype == torch.bfloat16 and sd["steps"].dtype == torch.int64
    # a bare state dict (stage2_slim) passes through
    n, meta = tool.extract_stage2_ema(
        str(tmp_path / "out.pt"), str(tmp_path / "out2.pt"), "ema", "keep"
    )
    assert n == 3 and meta == {}


def test_resolve_weights_dir_local(tmp_path):
    rae = pytest.importorskip("peal.generators.rae_diffusion_autoencoder")
    d = tmp_path / "w"
    d.mkdir()
    assert rae.resolve_weights_dir(str(d)) == str(d)
    with pytest.raises(FileNotFoundError):
        rae.resolve_weights_dir(str(tmp_path / "missing"))


def test_resolve_weights_dir_hf_cached(tmp_path, monkeypatch):
    rae = pytest.importorskip("peal.generators.rae_diffusion_autoencoder")
    monkeypatch.setenv("PEAL_HF_WEIGHTS_DIR", str(tmp_path))
    target = tmp_path / "org__repo"
    target.mkdir()
    for name in ("decoder.pt", "stats.pt", "stage2_ema.pt"):
        (target / name).write_bytes(b"x")
    # complete cache: no download attempted (huggingface_hub is not even called)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **kw: (_ for _ in ()).throw(AssertionError("should not download")),
    )
    assert rae.resolve_weights_dir("hf://org/repo@v1") == str(target)


def test_generator_uses_weights_dir(tmp_path):
    """With config.weights set, the generator takes decoder.pt / stats.pt and
    the stage-2 file from that folder instead of the run directory."""
    rae = pytest.importorskip("peal.generators.rae_diffusion_autoencoder")
    from types import SimpleNamespace

    w = tmp_path / "weights"
    w.mkdir()
    for name in ("decoder.pt", "stats.pt", "stage2_ema.pt"):
        (w / name).write_bytes(b"x")
    run = tmp_path / "run"
    (run / "stage1_assets").mkdir(parents=True)
    (run / "stage1_assets" / "decoder.pt").write_bytes(b"run")
    gen = object.__new__(rae.RAEDiffusionAutoencoder)
    gen.pipeline = SimpleNamespace(
        assets=str(run / "stage1_assets"), stage2_exp=str(run / "stage2" / "x")
    )
    gen.config = SimpleNamespace(weights=str(w), stage2_checkpoint=None)
    assert gen._asset("decoder.pt") == str(w / "decoder.pt")
    assert gen._asset("stats.pt") == str(w / "stats.pt")
    assert gen._find_stage2_checkpoint() == str(w / "stage2_ema.pt")
    # without weights: the run directory as before
    gen2 = object.__new__(rae.RAEDiffusionAutoencoder)
    gen2.pipeline = gen.pipeline
    gen2.config = SimpleNamespace(weights=None, stage2_checkpoint=None)
    assert gen2._asset("decoder.pt") == str(run / "stage1_assets" / "decoder.pt")
    assert gen2._asset("stats.pt") is None
    assert gen2._find_stage2_checkpoint() is None
