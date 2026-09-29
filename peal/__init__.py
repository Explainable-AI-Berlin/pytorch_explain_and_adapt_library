"""PEAL: the PyTorch Explain and Adapt Library.

PEAL explains image (and tabular) classifiers with counterfactuals and repairs
them from the feedback those explanations collect. The subpackages form one
pipeline: ``peal.data`` loads datasets, ``peal.architectures`` and
``peal.training`` build and train predictors, ``peal.generators`` hold the
generative models (diffusion autoencoders, DDPMs, Stable Diffusion, PathLDM,
Flux) that render counterfactuals, ``peal.explainers`` turn a predictor plus a
generator into explanations, ``peal.teachers`` label them (humans, LLMs, other
models, clustering), ``peal.adaptors`` (CFKD, DiDAE, ClArC, GroupDRO) use the
labels to repair the predictor and ``peal.sparse_dictionaries`` provide the
SAE / MSAE / Procrustes / SVD concept bases DiDAE edits along.
``peal.visualization`` and ``peal.web`` render figures and the demo. Every
component is configured through pydantic models loaded from yaml files.
"""

#: Single source of the distribution version. ``setup.py`` parses this file
#: textually (it cannot import the package before its dependencies exist) and
#: ``docs/conf.py`` reads the same attribute, so the number is written once.
__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Public API.
#
# Names below are the supported entry points; anything else under peal.* is an
# implementation detail that may move. They are resolved lazily through
# module __getattr__ so that `import peal` stays instant (it imports nothing
# but this file) while `peal.DiDAE` or `from peal import get_generator` work
# as if they had been imported eagerly. Importing a heavy name pulls in torch
# and the corresponding subpackage on first access only.
# ---------------------------------------------------------------------------

_API = {
    # configs and loading
    "load_yaml_config": "peal.global_utils",
    "save_yaml_config": "peal.global_utils",
    "set_random_seed": "peal.global_utils",
    "propagate_seed": "peal.global_utils",
    "DataConfig": "peal.data.interfaces",
    "TaskConfig": "peal.architectures.interfaces",
    "PredictorConfig": "peal.training.interfaces",
    "TrainingConfig": "peal.training.interfaces",
    "GeneratorConfig": "peal.generators.interfaces",
    "ExplainerConfig": "peal.explainers.interfaces",
    "AdaptorConfig": "peal.adaptors.interfaces",
    "SparseDictionaryConfig": "peal.sparse_dictionaries.interfaces",
    # factories
    "get_datasets": "peal.data.dataset_factory",
    "create_dataloaders_from_datasource": "peal.data.dataloaders",
    "get_predictor": "peal.architectures.predictors",
    "get_generator": "peal.generators.generator_factory",
    "get_explainer": "peal.explainers.explainer_factory",
    "get_teacher": "peal.teachers.teacher_factory",
    "get_adaptor": "peal.adaptors.adaptor_factory",
    "get_sparse_dictionary": "peal.sparse_dictionaries.sparse_dictionary_factory",
    # interfaces to implement
    "PealDataset": "peal.data.interfaces",
    "Generator": "peal.generators.interfaces",
    "InvertibleGenerator": "peal.generators.interfaces",
    "EditCapableGenerator": "peal.generators.interfaces",
    "ExplainerInterface": "peal.explainers.interfaces",
    "TeacherInterface": "peal.teachers.interfaces",
    "Adaptor": "peal.adaptors.interfaces",
    "SparseDictionary": "peal.sparse_dictionaries.interfaces",
    # the methods
    "ModelTrainer": "peal.training.trainers",
    "calculate_test_accuracy": "peal.training.trainers",
    "distill_predictor": "peal.training.trainers",
    "CFKD": "peal.adaptors.counterfactual_knowledge_distillation",
    "CFKDConfig": "peal.adaptors.counterfactual_knowledge_distillation",
    "DiDAE": "peal.adaptors.didae",
    "DiDAEConfig": "peal.adaptors.didae",
    # plumbing
    "get_logger": "peal.log",
    "registry": "peal.registry",
    "register": "peal.registry",
}

__all__ = sorted(_API) + ["__version__"]


def __getattr__(name):
    """Resolve a public name on first access (PEP 562)."""
    try:
        module_name = _API[name]
    except KeyError:
        raise AttributeError(f"module 'peal' has no attribute {name!r}") from None
    import importlib

    module = importlib.import_module(module_name)
    value = module if name == "registry" else getattr(module, name)
    globals()[name] = value  # cache: later accesses skip __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(_API))
