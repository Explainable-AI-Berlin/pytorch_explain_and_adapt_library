"""Logging replaces print, and the library no longer touches the caller's cwd."""

import os
import subprocess
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIRST_PARTY = os.path.join(ROOT, "peal")


def _first_party_sources():
    for root, dirs, files in os.walk(FIRST_PARTY):
        dirs[:] = [d for d in dirs if d not in ("dependencies", "__pycache__")]
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(root, f)


def test_get_logger_nests_under_peal_and_writes_bare_messages_to_stdout(capsys):
    from peal.log import get_logger

    log = get_logger("peal.some.module")
    assert log.name == "peal.some.module"
    assert get_logger("outsider").name == "peal.outsider"
    log.info("hello %s", "world")
    captured = capsys.readouterr()
    assert captured.out.strip().endswith("hello world") and captured.err == ""


def test_log_level_env_is_respected():
    code = (
        "import os; os.environ['PEAL_LOG_LEVEL']='WARNING'; "
        "from peal.log import get_logger; get_logger('peal.t').info('SHOULD_NOT_APPEAR'); "
        "get_logger('peal.t').warning('SHOULD_APPEAR')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert "SHOULD_APPEAR" in out and "SHOULD_NOT_APPEAR" not in out


def test_quiet_env_attaches_no_handler():
    code = (
        "import os, logging; os.environ['PEAL_LOG_QUIET']='1'; "
        "from peal.log import get_logger; get_logger('peal.t'); "
        "print(len(logging.getLogger('peal').handlers))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "0"


def test_no_relative_rmtree_of_static_anywhere():
    offenders = [
        p for p in _first_party_sources() if 'rmtree("static"' in open(p).read()
    ]
    assert not offenders, offenders


def test_no_unconditional_hf_login():
    src = open(
        os.path.join(FIRST_PARTY, "generators", "diffusion_autoencoder.py")
    ).read()
    assert "                login()  # login with" not in src
    assert "sys.stdin.isatty()" in src


def test_trainer_disables_grad_outside_training():
    src = open(os.path.join(FIRST_PARTY, "training", "trainers.py")).read()
    assert 'with torch.set_grad_enabled(mode == "train"):' in src
    assert 'if mode == "train":\n                loss.backward()' in src


def test_dataloader_workers_default_to_zero_but_are_configurable(monkeypatch):
    from types import SimpleNamespace

    from peal.data.dataloaders import resolve_num_workers

    monkeypatch.delenv("PEAL_NUM_WORKERS", raising=False)
    assert resolve_num_workers(None) == 0
    assert resolve_num_workers(SimpleNamespace(num_workers=0)) == 0
    assert resolve_num_workers(SimpleNamespace(num_workers=4)) == 4
    monkeypatch.setenv("PEAL_NUM_WORKERS", "2")
    assert resolve_num_workers(SimpleNamespace(num_workers=0)) == 2
    assert resolve_num_workers(SimpleNamespace(num_workers=6)) == 6
