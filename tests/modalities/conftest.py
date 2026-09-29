"""Fixtures of the modality tests.

Three guarantees hold for every test in this directory:

* the reproduction surface (``reproduction_scripts/``, ``configs/unit_test/``,
  the root entry-point wrappers, ``setup.py``) is byte-for-byte untouched when
  the session ends -- a session fixture snapshots it and fails otherwise;
* ``$PEAL_RUNS`` / ``$PEAL_DATA`` point into the test's temporary directory
  and the working directory is that temporary directory, so nothing a
  component writes can land in the checkout;
* the toy components are registered through ``monkeypatch`` and disappear
  from ``peal.registry`` again after the test.
"""

import glob
import os

import pytest

from tests.modalities import toys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROTECTED = ["reproduction_scripts", os.path.join("configs", "unit_test"), "setup.py"]
PROTECTED += sorted(
    os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "*.py"))
)


def _snapshot():
    """``{relative path: (size, mtime_ns)}`` of every protected file."""
    state = {}
    for entry in PROTECTED:
        path = os.path.join(ROOT, entry)
        files = [path] if os.path.isfile(path) else []
        for folder, _, names in os.walk(path):
            files += [os.path.join(folder, n) for n in names]
        for f in files:
            stat = os.stat(f)
            state[os.path.relpath(f, ROOT)] = (stat.st_size, stat.st_mtime_ns)
    return state


@pytest.fixture(scope="session", autouse=True)
def reproduction_surface_is_untouched():
    before = _snapshot()
    yield
    after = _snapshot()
    changed = sorted(
        set(before) ^ set(after) | {p for p in before if before[p] != after.get(p)}
    )
    assert not changed, f"a modality test touched the reproduction surface: {changed}"


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("PEAL_RUNS", str(tmp_path / "peal_runs"))
    monkeypatch.setenv("PEAL_DATA", str(tmp_path / "datasets"))
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def registered_toys(monkeypatch):
    """Register the toy classes the way a plugin would, for one test."""
    from peal.registry import REGISTRIES

    for kind, table in toys.REGISTRATIONS.items():
        for name, target in table.items():
            monkeypatch.setitem(REGISTRIES[kind], name, target)
    return toys.DOMAINS


@pytest.fixture(params=sorted(toys.DOMAINS))
def domain(request):
    return toys.DOMAINS[request.param]


@pytest.fixture
def datasets(domain, registered_toys):
    """``(train, val, test)`` of the domain, built by PEAL's dataset factory."""
    from peal.data.dataset_factory import get_datasets

    return get_datasets(toys.data_config(domain))
