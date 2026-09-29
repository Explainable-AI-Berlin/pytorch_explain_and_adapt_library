"""The contracts a new modality has to satisfy, checked on graphs, text and proteins.

Every test runs once per domain. A test that passes here means the seam it
exercises (registry, dataset factory, dataloaders, predictor, generator,
explainer record, teacher, serialisation round trip, sparse dictionary) is
modality-agnostic today; a strict ``xfail`` marks a seam that is still
image-only and names the refactor item that should flip it.
"""

import os
import types

import pytest
import torch

from tests.modalities import toys


# --------------------------------------------------------------------------
# registry and factories
# --------------------------------------------------------------------------


def test_registry_resolves_the_registered_classes(domain, registered_toys):
    from peal.registry import lookup

    assert lookup("datasets", domain.dataset.__name__) is domain.dataset
    assert lookup("generators", domain.generator.__name__) is domain.generator
    assert lookup("explainers", "ToyExplainer") is toys.ToyExplainer


def test_registration_does_not_leak_between_tests(domain):
    from peal.registry import REGISTRIES

    assert domain.dataset.__name__ not in REGISTRIES["datasets"]
    assert domain.generator.__name__ not in REGISTRIES["generators"]


def test_dataset_factory_builds_a_registered_modality(domain, datasets):
    train, val, test = datasets
    assert all(isinstance(d, domain.dataset) for d in (train, val, test))
    assert (len(train), len(val), len(test)) == (72, 12, 12)
    x, y = train[0]
    assert list(x.shape) == domain.dataset.input_size
    assert isinstance(y, int) and y in (0, 1)
    assert domain.dataset.is_valid(x)
    # the factory attaches an identity normalisation to non-image data
    assert torch.equal(train.normalization(x), x)


def test_unknown_modality_needs_a_dataset_class(domain, registered_toys):
    """Without ``dataset_class`` the input_type if-chain is the only route (A4)."""
    from peal.data.dataset_factory import get_datasets

    with pytest.raises(ValueError, match="not supported"):
        get_datasets(toys.data_config(domain, dataset_class=None))


def test_label_rule_is_balanced_enough_to_learn(domain, datasets):
    train = datasets[0]
    assert 0.25 <= float(train.y.float().mean()) <= 0.75
    assert torch.equal(domain.rule(train.x), train.y)


# --------------------------------------------------------------------------
# dataloaders
# --------------------------------------------------------------------------


def test_dataloaders_batch_the_domain_tensor(domain, datasets):
    from peal.architectures.interfaces import TaskConfig
    from peal.data.dataloaders import create_dataloaders_from_datasource
    from peal.training.interfaces import TrainingConfig

    config = types.SimpleNamespace(
        data=toys.data_config(domain),
        training=TrainingConfig(
            train_batch_size=8, val_batch_size=4, test_batch_size=4
        ),
        task=TaskConfig(output_channels=2),
    )
    train_loader, val_loader, test_loader = create_dataloaders_from_datasource(
        config, datasource=datasets
    )
    x, y = next(iter(train_loader))
    assert list(x.shape) == [8] + domain.dataset.input_size
    assert x.dtype == datasets[0][0][0].dtype  # long ids stay long
    assert y.dtype == torch.long and y.shape == (8,)
    assert len(test_loader) == 3


def test_class_ordered_stack_pops_per_class(domain, datasets):
    from peal.data.dataloaders import DataStack

    stack = DataStack(datasets[0], num_classes=2)
    x0, y0 = stack.pop(0)
    x1, y1 = stack.pop(1)
    assert (y0, y1) == (0, 1)
    assert domain.dataset.is_valid(x0) and domain.dataset.is_valid(x1)


# --------------------------------------------------------------------------
# predictors
# --------------------------------------------------------------------------


def test_predictor_contract_and_evaluation(domain, datasets):
    from peal.data.dataloaders import get_dataloader
    from peal.training.trainers import calculate_test_accuracy

    train, _, test = datasets
    model = toys.quick_fit(domain.predictor(), train)
    x = torch.stack([test[i][0] for i in range(4)])
    logits = model(x)
    assert logits.shape == (4, 2)
    assert model.encode(x).dim() == 2  # a feature vector per sample
    loader = get_dataloader(test, batch_size=4, mode="test")
    accuracy = calculate_test_accuracy(model, loader, "cpu", tracking_level=0)
    assert 0.0 <= accuracy <= 1.0
    (
        accuracy_again,
        group_accuracies,
        group_distribution,
        _,
        worst,
    ) = calculate_test_accuracy(
        model, loader, "cpu", calculate_group_accuracies=True, tracking_level=0
    )
    assert accuracy_again == accuracy
    assert worst == min(group_accuracies)
    assert abs(sum(group_distribution) - 1.0) < 1e-6
    assert test.return_dict is False  # restored after the call


def test_a_fitted_predictor_learns_the_rule(domain, datasets):
    train, _, test = datasets
    model = toys.quick_fit(domain.predictor(), train)
    with torch.no_grad():
        accuracy = float((model(train.x).argmax(-1) == train.y).float().mean())
    assert accuracy > 0.75, f"{domain.name}: train accuracy {accuracy:.2f}"


# --------------------------------------------------------------------------
# generators
# --------------------------------------------------------------------------


def test_generator_implements_both_abcs(domain):
    from peal.generators.interfaces import EditCapableGenerator, InvertibleGenerator

    generator = domain.generator()
    assert not getattr(type(generator), "__abstractmethods__", set())
    assert isinstance(generator, InvertibleGenerator)
    assert isinstance(generator, EditCapableGenerator)


def test_generator_factory_builds_it_from_a_config_dict(domain, registered_toys):
    from peal.generators.generator_factory import get_generator

    generator = get_generator(
        {
            "config_name": "ToyGeneratorConfig",
            "generator_type": domain.generator.__name__,
            "category": "generator",
            "max_edits": 2,
        },
        device="cpu",
    )
    assert isinstance(generator, domain.generator)
    assert generator.config.max_edits == 2
    assert not generator.training  # the factory returns eval mode


def test_latent_round_trip_is_exact(domain, datasets):
    generator = domain.generator()
    x = torch.stack([datasets[0][i][0] for i in range(6)])
    z = generator.encode(x)
    assert z.shape[0] == 6
    assert torch.equal(generator.decode(z), x)
    samples = generator.sample_x(3)
    assert samples.shape[0] == 3
    assert all(domain.dataset.is_valid(s) for s in samples)


def test_edit_returns_the_explainer_record_and_stays_in_domain(domain, datasets):
    train = datasets[0]
    student = toys.quick_fit(domain.predictor(), train)
    generator = domain.generator()
    x = torch.stack([train[i][0] for i in range(8)])
    with torch.no_grad():
        source = student(x).argmax(-1)
    target = 1 - source
    start = torch.softmax(student(x), -1)[torch.arange(8), target]
    x_cf, z_diff, end, x_in, history, success = generator.edit(
        x_in=x,
        target_confidence_goal=0.6,
        source_classes=source,
        target_classes=target,
        predictor=student,
        explainer_config=toys.ToyExplainerConfig(max_edits=3),
        predictor_datasets=datasets,
    )
    assert len(x_cf) == len(z_diff) == len(end) == len(x_in) == 8
    assert history is None and success.dtype == torch.bool
    for i in range(8):
        assert domain.dataset.is_valid(x_cf[i])
        assert float(end[i]) >= float(start[i]) - 1e-6  # greedy never regresses
        assert float(z_diff[i].sum()) > 0 or torch.equal(x_cf[i], x[i])
        if success[i]:
            assert int(student(x_cf[i][None]).argmax(-1)) == int(target[i])
    assert bool(success.any()), f"{domain.name}: no counterfactual reached the goal"


# --------------------------------------------------------------------------
# explainer record -> teacher -> counterfactual dataset round trip
# --------------------------------------------------------------------------


def test_explain_teach_serialize_reload(domain, datasets, registered_toys, tmp_path):
    """The explain-and-adapt loop's data path, without CFKD's image assumptions."""
    from peal.data.dataloaders import get_dataloader
    from peal.data.dataset_factory import get_datasets
    from peal.explainers.explainer_factory import get_explainer
    from peal.teachers.teacher_factory import get_teacher

    train, val, _ = datasets
    student = toys.quick_fit(domain.predictor(), train)
    explainer = get_explainer(
        {
            "config_name": "ToyExplainerConfig",
            "explainer_type": "ToyExplainer",
            "category": "explainer",
            "max_edits": 3,
        },
        device="cpu",
        predictor_datasets=datasets,
        predictor=student,
        generator=domain.generator(),
    )
    assert isinstance(explainer, toys.ToyExplainer)
    batch = next(iter(get_dataloader(val, batch_size=8, mode="val")))
    record = explainer.explain_batch(batch)
    for key in (
        "x_list",
        "x_counterfactual_list",
        "z_difference_list",
        "y_target_end_confidence_list",
        "y_source_list",
        "y_target_list",
    ):
        assert len(record[key]) == 8, key

    teacher = get_teacher(
        toys.ToyOracle(domain.rule),
        output_size=2,
        adaptor_config=types.SimpleNamespace(),
        dataset=train,
        device="cpu",
    )
    feedback = teacher.get_feedback(
        x_counterfactual_list=record["x_counterfactual_list"],
        y_source_list=record["y_source_list"],
        x_list=record["x_list"],
        y_list=record["y_list"],
        y_target_end_confidence_list=record["y_target_end_confidence_list"],
        y_target_list=record["y_target_list"],
        student=student,
        base_dir=str(tmp_path / "feedback"),
    )
    assert len(feedback) == 8
    assert all(isinstance(verdict, str) for verdict in feedback)
    flipped = [i for i, s in enumerate(record["success_list"]) if s]
    for i in flipped:
        # 1-sided teaching: a flip of a sample the student already got wrong
        # is not judged; every other flip is judged against the oracle's rule
        if record["y_source_list"][i] != record["y_list"][i]:
            assert feedback[i] == "student originally wrong!"
        else:
            assert feedback[i] in ("true", "false")

    # the modality's own rendering of the explanations, in place of collages
    paths, _ = train.generate_contrastive_collage(
        record["x_list"],
        record["x_counterfactual_list"],
        record["y_target_list"],
        record["y_source_list"],
        record["y_list"],
        record["y_target_start_confidence_list"],
        record["y_target_end_confidence_list"],
        str(tmp_path / "collages"),
    )
    assert len(paths) == 8 and all(os.path.getsize(p) > 0 for p in paths)

    # the confirmed counterfactuals become a dataset PEAL can reload (A7)
    keep = [i for i in flipped if feedback[i] == "true"] or flipped[:1]
    cf_dir = str(tmp_path / "counterfactuals")
    train.serialize_dataset(
        cf_dir,
        [record["x_counterfactual_list"][i] for i in keep],
        [record["y_target_list"][i] for i in keep],
    )
    reloaded = get_datasets(
        toys.data_config(domain, dataset_path=cf_dir, split=[1.0, 1.0])
    )[0]
    assert len(reloaded) == len(keep)
    for j, i in enumerate(keep):
        x, y = reloaded[j]
        assert torch.equal(x, record["x_counterfactual_list"][i])
        assert y == record["y_target_list"][i]


# --------------------------------------------------------------------------
# sparse dictionaries on the predictor's feature space
# --------------------------------------------------------------------------


def test_svd_dictionary_fits_domain_embeddings(domain, datasets, tmp_path):
    from peal.sparse_dictionaries.singular_value_decomposition import (
        SVDDictionary,
        SVDDictionaryConfig,
    )

    train = datasets[0]
    model = toys.quick_fit(domain.predictor(), train)
    with torch.no_grad():
        embeddings = model.encode(train.x)
    mu = embeddings.mean(0)
    dictionary = SVDDictionary(
        SVDDictionaryConfig(n_components=4, act_size=embeddings.shape[1])
    )
    dictionary.fit(embeddings - mu)
    components = dictionary.get_components()
    assert components.shape == (embeddings.shape[1], 4)
    gram = components.T @ components
    assert torch.allclose(gram, torch.eye(4), atol=1e-4)
    codes = (embeddings - mu) @ components
    assert codes.shape == (len(train), 4)
    path = str(tmp_path / "svd.pt")
    dictionary.save_on_disk(path)
    other = SVDDictionary(SVDDictionaryConfig(n_components=4))
    other.load_from_disk(path)
    assert torch.equal(other.get_components(), components)


# --------------------------------------------------------------------------
# known image-only seams (strict xfails: they turn red when the seam is fixed)
# --------------------------------------------------------------------------


def test_modality_datasets_override_the_projection_hooks(domain, datasets):
    """A modality provides its own ``project_to/from_pytorch_default``.

    The pair bridges the predictor's input format and the generator's
    format; for these discrete domains both are the same tensor, so the
    overrides are identities and the generator round trip holds through them.
    """
    train = datasets[0]
    x = torch.stack([train[i][0] for i in range(4)])
    assert torch.equal(train.project_from_pytorch_default(x), x)
    assert torch.equal(train.project_to_pytorch_default(x), x)
    generator = domain.generator()
    z = generator.encode(train.project_to_pytorch_default(x))
    assert torch.equal(train.project_from_pytorch_default(generator.decode(z)), x)


def test_inheriting_the_base_projection_default_is_not_safe(domain, datasets, request):
    """Tier A2 of the modality screening, as a strict xfail.

    A dataset that forgets the override above inherits the base
    ``PealDataset.project_from_pytorch_default``, which resizes anything whose
    last three dims differ from ``input_size``: the graph tensor gets rescaled,
    the token ids raise in ``Resize([])``. The protein tensor ``[20, 16]``
    survives only by coincidence (``Resize(16)`` leaves a 20x16 "image" alone)
    and is asserted plainly. When the base becomes the identity and
    ``ImageDataset`` carries the resize, the two xfails turn into XPASS
    failures and the marker below should be removed.
    """
    from peal.data.interfaces import PealDataset

    if domain.name != "protein":
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True,
                reason="Tier A2: the base project_from_pytorch_default is image-only",
            )
        )
    train = datasets[0]
    x = torch.stack([train[i][0] for i in range(4)])
    out = PealDataset.project_from_pytorch_default(train, x)
    assert torch.equal(out, x)
