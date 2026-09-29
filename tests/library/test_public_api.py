"""`import peal` is instant and every public name resolves on first access."""

import subprocess
import sys

import pytest

import peal


def test_import_peal_pulls_in_no_heavy_modules():
    code = (
        "import sys, peal; "
        "heavy = [m for m in ('torch','transformers','matplotlib','pandas','sklearn') if m in sys.modules]; "
        "print(heavy)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "[]", f"import peal imported heavy modules: {out}"


def test_all_names_are_declared_and_dir_lists_them():
    assert (
        "DiDAE" in peal.__all__
        and "get_generator" in peal.__all__
        and "__version__" in peal.__all__
    )
    assert set(peal.__all__) <= set(dir(peal))


@pytest.mark.parametrize("name", [n for n in peal.__all__ if n != "__version__"])
def test_public_name_resolves(name):
    value = getattr(peal, name)
    assert value is not None
    # second access is served from the module namespace, not __getattr__
    assert name in vars(peal)


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError):
        peal.definitely_not_an_api  # noqa: B018
