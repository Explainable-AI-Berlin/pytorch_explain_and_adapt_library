"""Text-specific behaviour: discrete token edits that respect padding and vocabulary."""

import torch

from tests.modalities import toys
from tests.modalities.toys import PAD, SEQ_LEN, VOCAB, keyword_count


def test_sequences_are_padded_at_the_tail_only():
    dataset = toys.ToyTextDataset(config=toys.data_config(toys.DOMAINS["text"]))
    assert dataset.x.dtype == torch.long
    assert all(toys.ToyTextDataset.is_valid(x) for x in dataset.x)
    lengths = (dataset.x != PAD).sum(-1)
    assert int(lengths.min()) >= 6 and int(lengths.max()) <= SEQ_LEN


def test_classifier_ignores_padding():
    torch.manual_seed(0)
    model = toys.ToyTextClassifier().eval()
    x = torch.randint(1, VOCAB, (4, SEQ_LEN))
    shorter = x.clone()
    shorter[:, 8:] = PAD
    longer = shorter.clone()
    longer[:, 8:] = 3  # extra real tokens change the pooled embedding ...
    with torch.no_grad():
        assert not torch.allclose(model(shorter), model(longer))
        # ... but the padded positions themselves contribute nothing
        assert torch.allclose(
            model.encode(shorter),
            model.encode(x[:, :8]).new_tensor(model.encode(shorter)),
        )
        assert torch.allclose(model.encode(shorter), model.encode(x[:, :8]))


def test_edits_substitute_at_most_max_edits_real_tokens():
    domain = toys.DOMAINS["text"]
    dataset = toys.ToyTextDataset(config=toys.data_config(domain))
    student = toys.quick_fit(domain.predictor(), dataset)
    x = dataset.x[:12]
    with torch.no_grad():
        source = student(x).argmax(-1)
    x_cf, z_diff, end, *_ = toys.ToyTextGenerator().edit(
        x_in=x,
        target_confidence_goal=0.9,
        source_classes=source,
        target_classes=1 - source,
        predictor=student,
        explainer_config=toys.ToyExplainerConfig(max_edits=2),
    )
    for original, edited, mask in zip(x, x_cf, z_diff):
        assert toys.ToyTextDataset.is_valid(edited)
        changed = (edited != original).float()
        assert torch.equal(changed, mask)
        assert int(mask.sum()) <= 2
        assert torch.equal(edited == PAD, original == PAD)  # padding untouched


def test_edits_towards_the_positive_class_add_keywords():
    domain = toys.DOMAINS["text"]
    dataset = toys.ToyTextDataset(config=toys.data_config(domain))
    student = toys.quick_fit(domain.predictor(), dataset)
    negatives = dataset.x[dataset.y == 0][:10]
    x_cf, *_ = toys.ToyTextGenerator().edit(
        x_in=negatives,
        target_confidence_goal=0.9,
        source_classes=torch.zeros(len(negatives), dtype=torch.long),
        target_classes=torch.ones(len(negatives), dtype=torch.long),
        predictor=student,
        explainer_config=toys.ToyExplainerConfig(max_edits=2),
    )
    before = keyword_count(negatives)
    after = keyword_count(torch.stack(x_cf))
    assert int((after > before).sum()) >= len(negatives) // 2
