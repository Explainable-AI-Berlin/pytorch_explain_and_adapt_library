"""The explicit component registry replaces the directory scan as first lookup."""

import ast
import os

import pytest

from peal.registry import REGISTRIES, UnknownComponentError, lookup, register, resolve

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _class_defined_in(module_path, cls):
    """True when ``cls`` is a top-level class of the module file (no import)."""
    path = os.path.join(ROOT, module_path.replace(".", "/") + ".py")
    if not os.path.isfile(path):
        return False, path
    tree = ast.parse(open(path).read())
    return any(isinstance(n, ast.ClassDef) and n.name == cls for n in tree.body), path


@pytest.mark.parametrize("kind", sorted(REGISTRIES))
def test_every_entry_points_at_a_real_class(kind):
    """Checked statically so optional dependencies (captum, omegaconf) do not
    turn a registry typo into a skipped test."""
    bad = []
    for name, target in REGISTRIES[kind].items():
        module_path, _, cls = target.partition(":")
        ok, path = _class_defined_in(module_path, cls)
        if not ok:
            bad.append((name, target, path))
    assert not bad, f"registry {kind}: entries without a matching class: {bad}"


def test_lookup_registered_name_imports_lazily():
    cls = lookup("configs", "DataConfig")
    assert cls.__name__ == "DataConfig"
    cls = lookup("adaptors", "DiDAE")
    assert cls.__name__ == "DiDAE"


def test_unknown_name_raises_a_clear_error_not_unboundlocal():
    with pytest.raises(UnknownComponentError) as exc:
        lookup("adaptors", "NoSuchAdaptor")
    msg = str(exc.value)
    assert "NoSuchAdaptor" in msg and "DiDAE" in msg and "register" in msg
    # both exception families callers may already catch
    assert isinstance(exc.value, KeyError) and isinstance(exc.value, ValueError)


def test_register_adds_a_class_and_a_string_target(tmp_path):
    class Custom:
        pass

    register("adaptors", "_TestCustom", Custom)
    assert lookup("adaptors", "_TestCustom") is Custom
    register("adaptors", "_TestString", "peal.adaptors.didae:DiDAE")
    assert lookup("adaptors", "_TestString").__name__ == "DiDAE"
    del REGISTRIES["adaptors"]["_TestCustom"], REGISTRIES["adaptors"]["_TestString"]
    with pytest.raises(UnknownComponentError):
        register("nope", "x", Custom)


def test_scan_fallback_finds_unregistered_in_tree_class():
    """A class dropped into the tree without a registry entry still resolves."""
    from peal.adaptors.interfaces import Adaptor

    name = "CFKD"
    saved = REGISTRIES["adaptors"].pop(name)
    try:
        cls = lookup(
            "adaptors",
            name,
            base_class=Adaptor,
            scan_dir=os.path.join(ROOT, "peal", "adaptors"),
        )
        assert cls.__name__ == name
    finally:
        REGISTRIES["adaptors"][name] = saved


def test_factories_reject_unknown_types_with_the_registry_error():
    from peal.adaptors.adaptor_factory import get_adaptor
    from peal.sparse_dictionaries.sparse_dictionary_factory import get_sparse_dictionary

    with pytest.raises(UnknownComponentError):
        get_adaptor({"adaptor_type": "NoSuchAdaptor", "category": "adaptor"})
    with pytest.raises(UnknownComponentError):
        get_sparse_dictionary(
            {
                "sparse_dictionaries_type": "NoSuchDict",
                "category": "sparse_dictionaries",
            }
        )


def test_legacy_explainer_names_map_to_the_counterfactual_explainer():
    for legacy in ("SCE", "ACE", "TIME", "DAEdistill"):
        assert (
            resolve(REGISTRIES["explainers"][legacy]).__name__
            == "CounterfactualExplainer"
        )


def test_config_resolution_no_longer_scans_the_package(monkeypatch):
    """get_config_model must answer from the registry without find_subclasses."""
    import peal.global_utils as gu

    monkeypatch.setattr(
        gu, "find_subclasses", lambda *a, **k: pytest.fail("scan was used")
    )
    assert gu.get_config_model({"config_name": "DataConfig"}).__name__ == "DataConfig"
    assert (
        gu.get_config_model({"category": "adaptor", "adaptor_type": "DiDAE"}).__name__
        == "DiDAEConfig"
    )
