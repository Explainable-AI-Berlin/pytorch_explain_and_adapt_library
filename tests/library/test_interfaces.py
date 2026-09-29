"""Interfaces are real ABCs, and every shipped concrete class satisfies them."""

import importlib
import inspect

import pytest

from peal.registry import REGISTRIES, resolve

BASES = {
    "generators": ("peal.generators.interfaces", "Generator"),
    "adaptors": ("peal.adaptors.interfaces", "Adaptor"),
    "explainers": ("peal.explainers.interfaces", "ExplainerInterface"),
    "sparse_dictionaries": ("peal.sparse_dictionaries.interfaces", "SparseDictionary"),
}


@pytest.mark.parametrize("kind", sorted(BASES))
def test_interface_is_abstract(kind):
    mod, cls = BASES[kind]
    base = getattr(importlib.import_module(mod), cls)
    assert inspect.isabstract(base), f"{cls} declares no abstract methods"


@pytest.mark.parametrize(
    "kind,name",
    [(k, n) for k in BASES for n in sorted(REGISTRIES[k])],
)
def test_every_registered_class_implements_the_abstract_methods(kind, name):
    try:
        cls = resolve(REGISTRIES[kind][name])
    except ImportError as exc:  # optional dependency (captum, omegaconf, ...)
        pytest.skip(f"{name}: optional dependency missing ({exc})")
    left = getattr(cls, "__abstractmethods__", set())
    assert not left, f"{name} leaves abstract methods unimplemented: {sorted(left)}"


def test_teacher_interface_raises_not_implemented():
    from peal.teachers.interfaces import TeacherInterface

    assert inspect.isabstract(TeacherInterface)
    with pytest.raises(TypeError):
        TeacherInterface()
