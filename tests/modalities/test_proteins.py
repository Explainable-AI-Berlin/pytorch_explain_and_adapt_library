"""Protein-specific behaviour: one-hot residues, point mutations, the hydrophobicity rule."""

import torch

from tests.modalities import toys
from tests.modalities.toys import (
    AMINO_ACIDS,
    N_AMINO,
    PROTEIN_LEN,
    hydrophobic_fraction,
)


def test_sequences_are_valid_one_hot_and_describe_round_trips():
    dataset = toys.ToyProteinDataset(config=toys.data_config(toys.DOMAINS["protein"]))
    assert dataset.x.shape[1:] == (N_AMINO, PROTEIN_LEN)
    assert all(toys.ToyProteinDataset.is_valid(x) for x in dataset.x)
    sequence = toys.ToyProteinDataset.describe(dataset.x[0])
    assert len(sequence) == PROTEIN_LEN and set(sequence) <= set(AMINO_ACIDS)


def test_hydrophobicity_rule_matches_the_letters():
    dataset = toys.ToyProteinDataset(config=toys.data_config(toys.DOMAINS["protein"]))
    for x, y in zip(dataset.x[:20], dataset.y[:20]):
        letters = toys.ToyProteinDataset.describe(x)
        fraction = sum(c in "AVILMFWY" for c in letters) / PROTEIN_LEN
        assert abs(float(hydrophobic_fraction(x)) - fraction) < 1e-6
        assert int(y) == int(fraction > 0.4)


def test_edits_are_point_mutations():
    domain = toys.DOMAINS["protein"]
    dataset = toys.ToyProteinDataset(config=toys.data_config(domain))
    student = toys.quick_fit(domain.predictor(), dataset)
    x = dataset.x[:12]
    with torch.no_grad():
        source = student(x).argmax(-1)
    x_cf, z_diff, *_ = toys.ToyProteinGenerator().edit(
        x_in=x,
        target_confidence_goal=0.9,
        source_classes=source,
        target_classes=1 - source,
        predictor=student,
        explainer_config=toys.ToyExplainerConfig(max_edits=3),
    )
    for original, edited, mask in zip(x, x_cf, z_diff):
        assert toys.ToyProteinDataset.is_valid(edited)
        mutated = (edited.argmax(-2) != original.argmax(-2)).float()
        assert torch.equal(mutated, mask)
        assert int(mask.sum()) <= 3


def test_edits_towards_hydrophobic_raise_the_fraction():
    domain = toys.DOMAINS["protein"]
    dataset = toys.ToyProteinDataset(config=toys.data_config(domain))
    student = toys.quick_fit(domain.predictor(), dataset)
    negatives = dataset.x[dataset.y == 0][:10]
    x_cf, *_ = toys.ToyProteinGenerator().edit(
        x_in=negatives,
        target_confidence_goal=0.9,
        source_classes=torch.zeros(len(negatives), dtype=torch.long),
        target_classes=torch.ones(len(negatives), dtype=torch.long),
        predictor=student,
        explainer_config=toys.ToyExplainerConfig(max_edits=3),
    )
    before = hydrophobic_fraction(negatives)
    after = hydrophobic_fraction(torch.stack(x_cf))
    assert int((after > before).sum()) >= len(negatives) // 2
