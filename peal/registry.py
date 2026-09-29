"""Explicit registries of PEAL's pluggable components.

Until 2026-09-25 every factory found its classes by importing *every* module
under a hard-coded ``peal/<subpackage>`` directory and keying the result by
class name (``peal.global_utils.find_subclasses``). That made out-of-tree
extension impossible, resolved duplicate class names by directory walk order,
imported every optional dependency on every config load, and cost about ten
seconds per lookup. These tables replace the scan as the first lookup; the scan
remains as a fallback, so a class dropped into the tree without a registry
entry is still found.

Values are ``"module.path:ClassName"`` and are imported lazily, so registering
a component costs nothing until it is used and a missing optional dependency
only matters when that component is requested.

To add a component from outside PEAL, call :func:`register` before the factory
runs, or add an entry point to the ``peal.components`` group in your own
package's metadata::

    [project.entry-points."peal.components"]
    generators = "mypkg.registry:GENERATORS"

where the target is a dict of the same ``name -> "module:Class"`` form.
"""

import importlib
import importlib.metadata
import os
import warnings

__all__ = [
    "GENERATORS",
    "ADAPTORS",
    "EXPLAINERS",
    "SPARSE_DICTIONARIES",
    "DATASETS",
    "CONFIGS",
    "REGISTRIES",
    "UnknownComponentError",
    "register",
    "resolve",
    "lookup",
]

GENERATORS = {
    "DDPM": "peal.generators.ddpm_generator:DDPM",
    "DDPMPathLDM": "peal.generators.ddpm_pathldm:DDPMPathLDM",
    "DiffusionAutoencoder": "peal.generators.diffusion_autoencoder:DiffusionAutoencoder",
    "DiffusionGenerator": "peal.generators.stable_diffusion_3:DiffusionGenerator",
    "FluxGenerator": "peal.generators.flux_1:FluxGenerator",
    "PathldmAutoencoder": "peal.generators.pathldm_autoencoder:PathldmAutoencoder",
    "RAEDiffusionAutoencoder": "peal.generators.rae_diffusion_autoencoder:RAEDiffusionAutoencoder",
    "StableDiffusion": "peal.generators.stable_diffusion_generator:StableDiffusion",
    "StableDiffusionAutoencoder": "peal.generators.stable_diffusion_autoencoder:StableDiffusionAutoencoder",
    "TabularDDPM": "peal.generators.tabular_ddpm:TabularDDPM",
}

ADAPTORS = {
    "CFKD": "peal.adaptors.counterfactual_knowledge_distillation:CFKD",
    "ClArC": "peal.adaptors.clarc:ClArC",
    "DiDAE": "peal.adaptors.didae:DiDAE",
    "GroupDRO": "peal.adaptors.group_distributionally_robust_optimization:GroupDRO",
    "GroupDROv2": "peal.adaptors.group_dro_v2:GroupDROv2",
    "PClArC": "peal.adaptors.clarc:PClArC",
    "ProjectionAdaptor": "peal.adaptors.projection_adaptor:ProjectionAdaptor",
    "RRClArC": "peal.adaptors.clarc:RRClArC",
}

EXPLAINERS = {
    "ACE": "peal.explainers.counterfactual_explainer:CounterfactualExplainer",
    "CounterfactualExplainer": "peal.explainers.counterfactual_explainer:CounterfactualExplainer",
    "DAEdistill": "peal.explainers.counterfactual_explainer:CounterfactualExplainer",
    "DiCEExplainer": "peal.explainers.no_generator_counterfactual_explainers:DiCEExplainer",
    "SCE": "peal.explainers.counterfactual_explainer:CounterfactualExplainer",
    "TIME": "peal.explainers.counterfactual_explainer:CounterfactualExplainer",
}

SPARSE_DICTIONARIES = {
    "BatchTopKSAE": "peal.sparse_dictionaries.batch_topk:BatchTopKSAE",
    "MSAEDecomposition": "peal.sparse_dictionaries.msae_decomposition:MSAEDecomposition",
    "OrthogonalProcrustesDictionary": "peal.sparse_dictionaries.orthogonal_procrustes_dictionary:OrthogonalProcrustesDictionary",
    "ProbeSAE": "peal.sparse_dictionaries.probe_sparse_autoencoder:ProbeSAE",
    "RASAEDecomposition": "peal.sparse_dictionaries.ra_sae_decomposition:RASAEDecomposition",
    "SVDDictionary": "peal.sparse_dictionaries.singular_value_decomposition:SVDDictionary",
    "SVDFilteredBatchTopKSAE": "peal.sparse_dictionaries.svd_filtered_sae:SVDFilteredBatchTopKSAE",
    "SpLICEDecomposition": "peal.sparse_dictionaries.splice_decomposition:SpLICEDecomposition",
}

DATASETS = {
    "AdultDataset": "peal.data.tabular_datasets:AdultDataset",
    "Camelyon17AugmentedDataset": "peal.data.custom_datasets:Camelyon17AugmentedDataset",
    "Camelyon17Dataset": "peal.data.custom_datasets:Camelyon17Dataset",
    "CelebACopyrighttagDataset": "peal.data.custom_datasets:CelebACopyrighttagDataset",
    "CelebADataset": "peal.data.custom_datasets:CelebADataset",
    "CelebAHQDataset": "peal.data.custom_datasets:CelebAHQDataset",
    "CircleDataset": "peal.data.tabular_datasets:CircleDataset",
    "ColoredMnist": "peal.data.custom_datasets:ColoredMnist",
    "CompassDataset": "peal.data.tabular_datasets:CompassDataset",
    "FollicleDataset": "peal.data.custom_datasets:FollicleDataset",
    "FunnyNodulesDataset": "peal.data.custom_datasets:FunnyNodulesDataset",
    "GermanDataset": "peal.data.tabular_datasets:GermanDataset",
    "ISICAnnotatedDataset": "peal.data.custom_datasets:ISICAnnotatedDataset",
    "Image2ClassDataset": "peal.data.datasets:Image2ClassDataset",
    "Image2MixedDataset": "peal.data.datasets:Image2MixedDataset",
    "ImageDataset": "peal.data.datasets:ImageDataset",
    "MnistDataset": "peal.data.custom_datasets:MnistDataset",
    "NicoPlusPlusDataset": "peal.data.custom_datasets:NicoPlusPlusDataset",
    "OnlySparseNumbersDataset": "peal.data.custom_datasets:OnlySparseNumbersDataset",
    "OnlySparseNumbersZipfDataset": "peal.data.custom_datasets:OnlySparseNumbersZipfDataset",
    "RxRx1AugmentedDataset": "peal.data.custom_datasets:RxRx1AugmentedDataset",
    "RxRx1Dataset": "peal.data.custom_datasets:RxRx1Dataset",
    "SkinConDataset": "peal.data.custom_datasets:SkinConDataset",
    "SparseNumbersDataset": "peal.data.custom_datasets:SparseNumbersDataset",
    "SparseNumbersDenseDataset": "peal.data.custom_datasets:SparseNumbersDenseDataset",
    "SparseNumbersDenseZipfDataset": "peal.data.custom_datasets:SparseNumbersDenseZipfDataset",
    "SparseNumbersZipfDataset": "peal.data.custom_datasets:SparseNumbersZipfDataset",
    "SquareDataset": "peal.data.custom_datasets:SquareDataset",
    "SymbolicDataset": "peal.data.datasets:SymbolicDataset",
    "WaterbirdsDataset": "peal.data.custom_datasets:WaterbirdsDataset",
}

CONFIGS = {
    "ACEConfig": "peal.explainers.counterfactual_explainer:ACEConfig",
    "AdaptorConfig": "peal.adaptors.interfaces:AdaptorConfig",
    "ArchitectureConfig": "peal.architectures.interfaces:ArchitectureConfig",
    "BatchTopKSAEConfig": "peal.sparse_dictionaries.batch_topk:BatchTopKSAEConfig",
    "CFKDConfig": "peal.adaptors.counterfactual_knowledge_distillation:CFKDConfig",
    "ClArCConfig": "peal.adaptors.clarc:ClArCConfig",
    "ColoredMnistConfig": "peal.data.custom_datasets:ColoredMnistConfig",
    "DAEdistillConfig": "peal.explainers.counterfactual_explainer:DAEdistillConfig",
    "DDPMConfig": "peal.generators.ddpm_generator:DDPMConfig",
    "DDPMInversionConfig": "peal.editors.ddpm_inversion:DDPMInversionConfig",
    "DDPMPathLDMConfig": "peal.generators.ddpm_pathldm:DDPMPathLDMConfig",
    "DataConfig": "peal.data.interfaces:DataConfig",
    "DiCEExplainerConfig": "peal.explainers.no_generator_counterfactual_explainers:DiCEExplainerConfig",
    "DiDAEConfig": "peal.adaptors.didae:DiDAEConfig",
    "DiffusionAutoencoderConfig": "peal.generators.diffusion_autoencoder:DiffusionAutoencoderConfig",
    "DiffusionGeneratorConfig": "peal.generators.stable_diffusion_3:DiffusionGeneratorConfig",
    "EditorConfig": "peal.editors.interfaces:EditorConfig",
    "ExplainerConfig": "peal.explainers.interfaces:ExplainerConfig",
    "FCConfig": "peal.architectures.interfaces:FCConfig",
    "GeneratorConfig": "peal.generators.interfaces:GeneratorConfig",
    "GroupDROConfig": "peal.adaptors.group_distributionally_robust_optimization:GroupDROConfig",
    "GroupDROv2Config": "peal.adaptors.group_dro_v2:GroupDROv2Config",
    "MSAEDecompositionConfig": "peal.sparse_dictionaries.msae_decomposition:MSAEDecompositionConfig",
    "OrthogonalProcrustesDictionaryConfig": "peal.sparse_dictionaries.orthogonal_procrustes_dictionary:OrthogonalProcrustesDictionaryConfig",
    "PClArCConfig": "peal.adaptors.clarc:PClArCConfig",
    "PathldmAutoencoderConfig": "peal.generators.pathldm_autoencoder:PathldmAutoencoderConfig",
    "PerfectFalseCounterfactualConfig": "peal.explainers.counterfactual_explainer:PerfectFalseCounterfactualConfig",
    "PredictorConfig": "peal.training.interfaces:PredictorConfig",
    "ProbeSAEConfig": "peal.sparse_dictionaries.probe_sparse_autoencoder:ProbeSAEConfig",
    "ProjectionAdaptorConfig": "peal.adaptors.projection_adaptor:ProjectionAdaptorConfig",
    "RAEDiffusionAutoencoderConfig": "peal.generators.rae_diffusion_autoencoder:RAEDiffusionAutoencoderConfig",
    "RASAEDecompositionConfig": "peal.sparse_dictionaries.ra_sae_decomposition:RASAEDecompositionConfig",
    "RRClArCConfig": "peal.adaptors.clarc:RRClArCConfig",
    "ResnetConfig": "peal.architectures.interfaces:ResnetConfig",
    "SCEConfig": "peal.explainers.counterfactual_explainer:SCEConfig",
    "SVDDictionaryConfig": "peal.sparse_dictionaries.singular_value_decomposition:SVDDictionaryConfig",
    "SVDFilteredBatchTopKSAEConfig": "peal.sparse_dictionaries.svd_filtered_sae:SVDFilteredBatchTopKSAEConfig",
    "SVDFilteredSAEConfig": "peal.sparse_dictionaries.svd_filtered_sae:SVDFilteredSAEConfig",
    "SpLICEDecompositionConfig": "peal.sparse_dictionaries.splice_decomposition:SpLICEDecompositionConfig",
    "SparseDictionaryConfig": "peal.sparse_dictionaries.interfaces:SparseDictionaryConfig",
    "SprayConfig": "peal.teachers.spray_teacher:SprayConfig",
    "StableDiffusionAutoencoderConfig": "peal.generators.stable_diffusion_autoencoder:StableDiffusionAutoencoderConfig",
    "StableDiffusionConfig": "peal.generators.stable_diffusion_generator:StableDiffusionConfig",
    "TIMEConfig": "peal.explainers.counterfactual_explainer:TIMEConfig",
    "TabularDDPMConfig": "peal.generators.tabular_ddpm:TabularDDPMConfig",
    "TaskConfig": "peal.architectures.interfaces:TaskConfig",
    "TrainingConfig": "peal.training.interfaces:TrainingConfig",
    "TransformerConfig": "peal.architectures.interfaces:TransformerConfig",
    "VGGConfig": "peal.architectures.interfaces:VGGConfig",
}


REGISTRIES = {
    "generators": GENERATORS,
    "adaptors": ADAPTORS,
    "explainers": EXPLAINERS,
    "sparse_dictionaries": SPARSE_DICTIONARIES,
    "datasets": DATASETS,
    "configs": CONFIGS,
}

_ENTRY_POINTS_LOADED = False


class UnknownComponentError(KeyError, ValueError):
    """A ``*_type`` string names no registered or discoverable class.

    Subclasses both ``KeyError`` and ``ValueError`` so callers that caught either
    keep working. It replaces the ``UnboundLocalError`` the factories used to
    raise from an unassigned ``*_out`` variable.
    """


def register(kind, name, target):
    """Add ``name -> target`` to the ``kind`` registry.

    Parameters
    ----------
    kind : str
        One of :data:`REGISTRIES`' keys.
    name : str
        The ``*_type`` string configs will use.
    target : str or type
        ``"module.path:ClassName"`` or the class itself.
    """
    if kind not in REGISTRIES:
        raise UnknownComponentError(
            f"unknown registry {kind!r}; one of {sorted(REGISTRIES)}"
        )
    # A class object is stored as is: turning it into "module:qualname" would
    # break for anything not importable by attribute path (a class defined
    # inside a function, a dynamically created one). resolve() returns
    # non-string targets unchanged.
    REGISTRIES[kind][name] = target


def _load_entry_points():
    """Merge registries advertised by installed packages (once per process)."""
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    try:
        eps = importlib.metadata.entry_points()
        group = (
            eps.select(group="peal.components")
            if hasattr(eps, "select")
            else eps.get("peal.components", [])
        )
    except Exception:  # metadata unavailable (zipapp, frozen build)
        return
    for ep in group:
        if ep.name not in REGISTRIES:
            continue
        try:
            table = ep.load()
        except Exception as exc:  # a broken plugin must not break PEAL
            warnings.warn(
                f"[peal.registry] could not load entry point {ep.name} from {ep.value}: {exc}",
                RuntimeWarning,
            )
            continue
        if isinstance(table, dict):
            REGISTRIES[ep.name].update(table)


def resolve(target):
    """Import ``"module.path:ClassName"`` (a class is returned unchanged)."""
    if not isinstance(target, str):
        return target
    module_name, _, attr = target.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def lookup(kind, name, base_class=None, scan_dir=None):
    """Return the class registered as ``name`` in the ``kind`` registry.

    Parameters
    ----------
    kind : str
        Registry key, e.g. ``"generators"``.
    name : str
        The ``*_type`` value from a config.
    base_class : type, optional
        With ``scan_dir``, enables the legacy directory scan as a fallback for
        classes that exist in the tree but were never registered.
    scan_dir : str, optional
        Directory for that fallback.

    Raises
    ------
    UnknownComponentError
        When ``name`` is neither registered nor discoverable.
    """
    _load_entry_points()
    table = REGISTRIES[kind]
    if name in table:
        return resolve(table[name])
    if base_class is not None and scan_dir is not None and os.path.isdir(scan_dir):
        from peal.global_utils import (
            find_subclasses,
        )  # heavy; only on the fallback path

        found = {cls.__name__: cls for cls in find_subclasses(base_class, scan_dir)}
        if name in found:
            return found[name]
    raise UnknownComponentError(
        f"unknown {kind[:-1] if kind.endswith('s') else kind} type {name!r}. "
        f"Known: {sorted(table)}. Register it with peal.registry.register("
        f"{kind!r}, {name!r}, 'module.path:ClassName') or a 'peal.components' entry point."
    )
