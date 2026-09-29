"""Graph-specific behaviour: node permutation invariance and edge-level edits."""

import torch

from tests.modalities import toys
from tests.modalities.toys import N_NODES, TRIU, adjacency, has_triangle


def _permute(x, perm):
    adj, feats = adjacency(x), toys.node_features(x)
    return torch.cat([adj[perm][:, perm], feats[perm]], -1)


def test_triangle_rule_is_permutation_invariant():
    dataset = toys.ToyGraphDataset(config=toys.data_config(toys.DOMAINS["graph"]))
    perm = torch.randperm(N_NODES, generator=torch.Generator().manual_seed(1))
    permuted = torch.stack([_permute(x, perm) for x in dataset.x])
    assert torch.equal(has_triangle(adjacency(permuted)), dataset.y.bool())


def test_gcn_is_permutation_invariant():
    torch.manual_seed(0)
    model = toys.ToyGCN().eval()
    dataset = toys.ToyGraphDataset(config=toys.data_config(toys.DOMAINS["graph"]))
    perm = torch.randperm(N_NODES, generator=torch.Generator().manual_seed(2))
    x = dataset.x[:8]
    permuted = torch.stack([_permute(graph, perm) for graph in x])
    with torch.no_grad():
        assert torch.allclose(model(x), model(permuted), atol=1e-5)


def test_edits_flip_whole_edges_and_keep_the_graph_simple():
    domain = toys.DOMAINS["graph"]
    dataset = toys.ToyGraphDataset(config=toys.data_config(domain))
    student = toys.quick_fit(domain.predictor(), dataset)
    x = dataset.x[:12]
    with torch.no_grad():
        source = student(x).argmax(-1)
    x_cf, z_diff, *_ = toys.ToyGraphGenerator().edit(
        x_in=x,
        target_confidence_goal=0.9,
        source_classes=source,
        target_classes=1 - source,
        predictor=student,
        explainer_config=toys.ToyExplainerConfig(max_edits=3),
    )
    for original, edited, mask in zip(x, x_cf, z_diff):
        assert toys.ToyGraphDataset.is_valid(edited)
        # node features are never touched, only the adjacency
        assert torch.equal(toys.node_features(edited), toys.node_features(original))
        flipped = (adjacency(edited) != adjacency(original)).float()
        assert torch.equal(flipped, mask)
        assert torch.equal(mask, mask.T)
        assert int(mask[TRIU[0], TRIU[1]].sum()) <= 3


def test_latent_edge_indicators_are_the_upper_triangle():
    generator = toys.ToyGraphGenerator()
    dataset = toys.ToyGraphDataset(config=toys.data_config(toys.DOMAINS["graph"]))
    x = dataset.x[:5]
    z = generator.encode(x)
    assert z.shape == (5, toys.N_EDGES + N_NODES * toys.N_FEATURES)
    assert torch.equal(z[:, : toys.N_EDGES], adjacency(x)[:, TRIU[0], TRIU[1]])
