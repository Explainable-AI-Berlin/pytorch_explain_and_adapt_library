"""Toy graph, text and protein components for PEAL's extension seams.

Nothing here models anything real. Each domain is the smallest object that
has the *shape* of the real thing -- a graph as an adjacency matrix plus node
features, a padded token sequence, a one-hot amino-acid sequence -- so the
tests can push non-image tensors through PEAL's registry, dataset factory,
dataloaders, predictor / generator / explainer contracts, the model-to-model
teacher, the trainer and a sparse dictionary. Everything is seeded and runs
on the CPU in well under a second per domain, so the assertions are exact.

The components subclass the public interfaces and nothing else. If a refactor
of those interfaces breaks them, it breaks every out-of-tree modality the same
way, which is what these tests are for.
"""

import os
import types

import torch
import torch.nn.functional as F
from torch import nn

from peal.data.interfaces import DataConfig, PealDataset
from peal.explainers.interfaces import ExplainerConfig, ExplainerInterface
from peal.generators.interfaces import (
    EditCapableGenerator,
    GeneratorConfig,
    InvertibleGenerator,
)

# ---------------------------------------------------------------------------
# the three domains
# ---------------------------------------------------------------------------

#: graphs: 6 nodes, 3 node features, 15 possible undirected edges
N_NODES, N_FEATURES = 6, 3
TRIU = torch.triu_indices(N_NODES, N_NODES, offset=1)
N_EDGES = TRIU.shape[1]

#: text: 32 token ids, 0 is padding, sequences of at most 12 tokens
VOCAB, PAD, SEQ_LEN = 32, 0, 12
KEYWORDS = (5, 6, 7, 8, 9)

#: proteins: the 20 amino acids as one-hot channels over 16 positions
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
N_AMINO = len(AMINO_ACIDS)
PROTEIN_LEN = 16
HYDROPHOBIC = torch.tensor([aa in "AVILMFWY" for aa in AMINO_ACIDS]).float()


def adjacency(x):
    """The ``[..., N, N]`` adjacency block of a graph tensor ``[..., N, N + F]``."""
    return x[..., :N_NODES]


def node_features(x):
    """The ``[..., N, F]`` feature block of a graph tensor."""
    return x[..., N_NODES:]


def has_triangle(adj):
    """Whether each graph contains a 3-clique (``trace(A^3) > 0``)."""
    a3 = adj @ adj @ adj
    return torch.diagonal(a3, dim1=-2, dim2=-1).sum(-1) > 0


def keyword_count(ids):
    """How many keyword tokens each sequence contains."""
    return torch.isin(ids, torch.tensor(KEYWORDS)).sum(-1)


def hydrophobic_fraction(x):
    """Fraction of hydrophobic residues of a one-hot protein ``[..., 20, L]``."""
    return (x * HYDROPHOBIC[:, None]).sum(-2).mean(-1)


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------


class ToyDataset(PealDataset):
    """In-memory dataset with PEAL's split, dict-mode and serialisation hooks.

    Subclasses set ``input_size`` and define ``synthesize`` (the data and its
    label rule), ``describe`` (a one-line text rendering that stands in for a
    collage) and ``is_valid`` (the domain constraint an edit must respect).
    The constructor signature is the one ``dataset_factory.get_datasets`` uses.
    When ``dataset_path`` holds a ``samples.pt`` written by
    ``serialize_dataset`` it is loaded instead of synthesising, which is the
    round trip CFKD relies on for its counterfactual datasets.
    """

    input_size = None
    modality = "toy"

    def __init__(
        self,
        root_dir=None,
        mode="train",
        config=None,
        transform=None,
        return_dict=False,
        data_dir=None,
        task_config=None,
        **kwargs,
    ):
        self.config = config
        self.mode = mode
        self.transform = transform
        self.task_config = task_config
        self.return_dict = return_dict
        self.hints_enabled = False
        self.idx_enabled = False
        self.groups_enabled = False
        path = root_dir or config.dataset_path
        blob = os.path.join(path, "samples.pt") if path else None
        if blob and os.path.isfile(blob):
            saved = torch.load(blob)
            x_all, y_all = saved["x"], saved["y"]
        else:
            x_all, y_all = self.synthesize(config.num_samples or 64, config.seed or 0)
        n = len(y_all)
        a, b = int(config.split[0] * n), int(config.split[1] * n)
        lo, hi = {"train": (0, a), "val": (a, b), "test": (b, n), "all": (0, n)}[mode]
        self._x_all, self._y_all = x_all[lo:hi], y_all[lo:hi]
        self.x, self.y = self._x_all, self._y_all
        n_inputs = int(torch.tensor(self.input_size).prod())
        self.attributes = [f"x{i}" for i in range(n_inputs)] + ["Target"]

    # -- torch Dataset protocol -------------------------------------------
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx].clone()
        y = int(self.y[idx])
        if self.return_dict:
            sample = {"x": x, "y": y}
            if self.idx_enabled:
                sample["index"] = idx
            if self.groups_enabled:
                sample["has_confounder"] = 0
            return sample
        if self.idx_enabled:
            return x, [y, idx]
        return x, y

    @property
    def output_size(self):
        if self.task_config is not None and self.task_config.output_channels:
            return int(self.task_config.output_channels)
        return int(self.config.output_size[0])

    # -- the switches trainers and teachers flip ---------------------------
    def enable_hints(self):
        self.hints_enabled = True

    def disable_hints(self):
        self.hints_enabled = False

    def enable_idx(self):
        self.idx_enabled = True

    def disable_idx(self):
        self.idx_enabled = False

    def enable_groups(self):
        self.groups_enabled = True

    def disable_groups(self):
        self.groups_enabled = False

    def enable_class_restriction(self, class_idx):
        keep = self._y_all == int(class_idx)
        self.x, self.y = self._x_all[keep], self._y_all[keep]

    def disable_class_restriction(self):
        self.x, self.y = self._x_all, self._y_all

    # -- hooks the teachers, explainers and adaptors call ------------------
    def calculate_outlier_score(self, x):
        zeros = torch.zeros(x.shape[0], device=x.device)
        return {"absolute": zeros, "relative": zeros}

    def project_to_pytorch_default(self, x):
        return x

    def project_from_pytorch_default(self, x):
        return x

    def serialize_dataset(
        self, output_dir, x_list, y_list, sample_names=None, **kwargs
    ):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(
            {
                "x": torch.stack([x.detach().cpu() for x in x_list]),
                "y": torch.tensor([int(y) for y in y_list]),
            },
            os.path.join(output_dir, "samples.pt"),
        )

    def generate_contrastive_collage(
        self,
        x_list,
        x_counterfactual_list,
        y_target_list,
        y_source_list,
        y_list,
        y_target_start_confidence_list,
        y_target_end_confidence_list,
        base_path,
        start_idx=0,
        **kwargs,
    ):
        """Text rendering of every pair: the modality's stand-in for a collage."""
        os.makedirs(base_path, exist_ok=True)
        paths = []
        for i, (x, x_cf) in enumerate(zip(x_list, x_counterfactual_list)):
            path = os.path.join(base_path, f"{start_idx + i}.txt")
            with open(path, "w") as handle:
                handle.write(
                    f"original ({int(y_source_list[i])} -> {int(y_target_list[i])}):\n"
                    f"  {self.describe(x)}\ncounterfactual:\n  {self.describe(x_cf)}\n"
                )
            paths.append(path)
        return paths, [None] * len(paths)

    # -- per modality -------------------------------------------------------
    def synthesize(self, n, seed):
        raise NotImplementedError

    @classmethod
    def describe(cls, x):
        raise NotImplementedError

    @classmethod
    def is_valid(cls, x):
        raise NotImplementedError


class ToyGraphDataset(ToyDataset):
    """Random graphs ``[N, N + F]`` (adjacency | node features), label = has a triangle."""

    modality = "graph"
    input_size = [N_NODES, N_NODES + N_FEATURES]

    def synthesize(self, n, seed):
        gen = torch.Generator().manual_seed(seed)
        edges = (torch.rand(n, N_EDGES, generator=gen) < 0.3).float()
        adj = torch.zeros(n, N_NODES, N_NODES)
        adj[:, TRIU[0], TRIU[1]] = edges
        adj = adj + adj.transpose(1, 2)
        feats = torch.randn(n, N_NODES, N_FEATURES, generator=gen)
        return torch.cat([adj, feats], dim=-1), has_triangle(adj).long()

    @classmethod
    def describe(cls, x):
        adj = adjacency(x)
        edges = [f"{i}-{j}" for i, j in zip(*TRIU.tolist()) if adj[i, j] > 0.5]
        return "edges: " + (" ".join(edges) or "none")

    @classmethod
    def is_valid(cls, x):
        adj = adjacency(x)
        binary = bool(((adj == 0) | (adj == 1)).all())
        symmetric = bool(torch.equal(adj, adj.transpose(-1, -2)))
        loop_free = bool((torch.diagonal(adj, dim1=-2, dim2=-1) == 0).all())
        return binary and symmetric and loop_free


class ToyTextDataset(ToyDataset):
    """Padded token-id sequences ``[L]`` (long), label = at least two keywords."""

    modality = "text"
    input_size = [SEQ_LEN]

    def synthesize(self, n, seed):
        gen = torch.Generator().manual_seed(seed)
        ids = torch.randint(1, VOCAB, (n, SEQ_LEN), generator=gen)
        lengths = torch.randint(6, SEQ_LEN + 1, (n,), generator=gen)
        ids[torch.arange(SEQ_LEN)[None, :] >= lengths[:, None]] = PAD
        return ids, (keyword_count(ids) >= 2).long()

    @classmethod
    def describe(cls, x):
        return " ".join(f"t{int(t)}" for t in x if int(t) != PAD)

    @classmethod
    def is_valid(cls, x):
        in_vocab = x.dtype == torch.long and bool(((x >= 0) & (x < VOCAB)).all())
        pads = (x == PAD).int()
        # padding only ever grows towards the end of the sequence
        tail_only = bool((pads[..., 1:] >= pads[..., :-1]).all())
        return in_vocab and tail_only


class ToyProteinDataset(ToyDataset):
    """One-hot amino-acid sequences ``[20, L]``, label = mostly hydrophobic."""

    modality = "protein"
    input_size = [N_AMINO, PROTEIN_LEN]

    def synthesize(self, n, seed):
        gen = torch.Generator().manual_seed(seed)
        ids = torch.randint(0, N_AMINO, (n, PROTEIN_LEN), generator=gen)
        x = F.one_hot(ids, N_AMINO).transpose(1, 2).float()
        return x, (hydrophobic_fraction(x) > 0.4).long()

    @classmethod
    def describe(cls, x):
        return "".join(AMINO_ACIDS[int(i)] for i in x.argmax(-2))

    @classmethod
    def is_valid(cls, x):
        one_hot = bool(((x == 0) | (x == 1)).all()) and bool((x.sum(-2) == 1).all())
        return x.dtype == torch.float32 and one_hot


# ---------------------------------------------------------------------------
# predictors: forward(x) -> logits, encode(x) -> features
# ---------------------------------------------------------------------------


class ToyGCN(nn.Module):
    """One-layer graph convolution with mean pooling."""

    def __init__(self, hidden=8, num_classes=2):
        super().__init__()
        self.lin = nn.Linear(N_FEATURES, hidden)
        self.out = nn.Linear(hidden, num_classes)

    def encode(self, x):
        a_hat = adjacency(x) + torch.eye(N_NODES, device=x.device)
        d = a_hat.sum(-1).clamp(min=1).rsqrt()
        a_norm = d[..., :, None] * a_hat * d[..., None, :]
        return torch.relu(a_norm @ self.lin(node_features(x))).mean(-2)

    def forward(self, x):
        return self.out(self.encode(x))


class ToyTextClassifier(nn.Module):
    """Embedding bag over the non-padding tokens."""

    def __init__(self, dim=8, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, dim, padding_idx=PAD)
        self.out = nn.Linear(dim, num_classes)

    def encode(self, x):
        mask = (x != PAD).float()[..., None]
        return (self.emb(x) * mask).sum(-2) / mask.sum(-2).clamp(min=1)

    def forward(self, x):
        return self.out(self.encode(x))


class ToyProteinCNN(nn.Module):
    """A width-3 convolution over the residues with global max pooling."""

    def __init__(self, hidden=8, num_classes=2):
        super().__init__()
        self.conv = nn.Conv1d(N_AMINO, hidden, kernel_size=3, padding=1)
        self.out = nn.Linear(hidden, num_classes)

    def encode(self, x):
        return torch.relu(self.conv(x)).amax(-1)

    def forward(self, x):
        return self.out(self.encode(x))


class ToyOracle(nn.Module):
    """A classifier that knows the label rule exactly (the "unpoisoned teacher")."""

    def __init__(self, rule):
        super().__init__()
        self.rule = rule
        self.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x):
        return F.one_hot(self.rule(x), 2).float() * 8.0 - 4.0


# ---------------------------------------------------------------------------
# generators: an explicit latent plus a greedy discrete edit
# ---------------------------------------------------------------------------


class ToyGeneratorConfig(GeneratorConfig):
    """Config of the toy generators (``max_edits`` bounds the greedy search)."""

    config_name: str = "ToyGeneratorConfig"
    generator_type: str = "ToyGraphGenerator"
    max_edits: int = 3


class _ToyDiscreteGenerator(InvertibleGenerator, EditCapableGenerator):
    """Shared machinery of the three toy generators.

    ``edit`` is a greedy search over single discrete moves (flip an edge,
    substitute a token, mutate a residue): at every step the move that most
    raises the predictor's confidence in the target class is applied, until
    the goal is reached, no move helps, or ``max_edits`` is spent. The
    "latent difference" it reports is the mask of edited positions, which is
    the discrete-edit contract a text or graph generator can offer where a
    diffusion model reports ``z - z'``. Subclasses provide ``encode`` /
    ``decode`` / ``sample_z`` and the move set ``proposals``.
    """

    def __init__(self, config=None, device=None, predictor_dataset=None, **kwargs):
        super().__init__()
        if config is None:
            config = ToyGeneratorConfig(generator_type=type(self).__name__)
        self.config = config
        self.predictor_dataset = predictor_dataset
        self.register_buffer("anchor", torch.zeros(1))

    def train_model(self):
        """Nothing to train: the toy latent is a fixed, exact code."""

    def sample_x(self, batch_size=1):
        return self.decode(self.sample_z(batch_size))

    @staticmethod
    def _confidence(predictor, x, target):
        return torch.softmax(predictor(x), -1)[:, target]

    @torch.no_grad()
    def edit(
        self,
        x_in,
        target_confidence_goal,
        source_classes,
        target_classes,
        predictor,
        explainer_config=None,
        predictor_datasets=None,
        boolmask_in=None,
        attempt_number=None,
        pbar=None,
        mode="",
        base_path="",
    ):
        max_edits = int(
            getattr(explainer_config, "max_edits", getattr(self.config, "max_edits", 3))
        )
        was_training = predictor.training
        predictor.eval()
        x_cf_list, diff_list, conf_list, success = [], [], [], []
        for x, target in zip(x_in, target_classes):
            x, target = x.clone(), int(target)
            changed = self.empty_change(x)
            conf = self._confidence(predictor, x[None], target)[0]
            for _ in range(max_edits):
                if conf >= target_confidence_goal:
                    break
                candidates, candidate_changes = self.proposals(x)
                confs = self._confidence(predictor, candidates, target)
                best = int(confs.argmax())
                if confs[best] <= conf:
                    break
                x, conf = candidates[best], confs[best]
                changed = torch.maximum(changed, candidate_changes[best])
            x_cf_list.append(x)
            diff_list.append(changed)
            conf_list.append(conf.clone())
            success.append(bool(conf >= target_confidence_goal))
        predictor.train(was_training)
        return (
            x_cf_list,
            diff_list,
            conf_list,
            list(x_in),
            None,
            torch.tensor(success),
        )

    def empty_change(self, x):
        raise NotImplementedError

    def proposals(self, x):
        raise NotImplementedError


class ToyGraphGenerator(_ToyDiscreteGenerator):
    """Latent = the 15 edge indicators followed by the flattened node features."""

    def encode(self, x, t=1.0, stochastic=None, num_steps=None):
        adj, feats = adjacency(x), node_features(x)
        return torch.cat([adj[..., TRIU[0], TRIU[1]], feats.flatten(-2)], -1)

    def decode(self, z, t=1.0, stochastic=None, num_steps=None):
        lead = z.shape[:-1]
        edges = (z[..., :N_EDGES] > 0.5).float()
        feats = z[..., N_EDGES:].reshape(*lead, N_NODES, N_FEATURES)
        adj = torch.zeros(*lead, N_NODES, N_NODES, device=z.device)
        adj[..., TRIU[0], TRIU[1]] = edges
        adj = adj + adj.transpose(-1, -2)
        return torch.cat([adj, feats], -1)

    def sample_z(self, batch_size=1):
        edges = (torch.rand(batch_size, N_EDGES) < 0.3).float()
        return torch.cat([edges, torch.randn(batch_size, N_NODES * N_FEATURES)], -1)

    def empty_change(self, x):
        return torch.zeros(N_NODES, N_NODES)

    def proposals(self, x):
        """Every single edge flip, keeping the adjacency symmetric."""
        candidates = x[None].repeat(N_EDGES, 1, 1)
        changes = torch.zeros(N_EDGES, N_NODES, N_NODES)
        for k, (i, j) in enumerate(zip(*TRIU.tolist())):
            flipped = 1.0 - candidates[k, i, j]
            candidates[k, i, j] = candidates[k, j, i] = flipped
            changes[k, i, j] = changes[k, j, i] = 1.0
        return candidates, changes


class ToyTextGenerator(_ToyDiscreteGenerator):
    """Latent = one-hot tokens ``[L, V]``; decoding takes the argmax."""

    def encode(self, x, t=1.0, stochastic=None, num_steps=None):
        return F.one_hot(x, VOCAB).float()

    def decode(self, z, t=1.0, stochastic=None, num_steps=None):
        return z.argmax(-1)

    def sample_z(self, batch_size=1):
        return F.one_hot(torch.randint(1, VOCAB, (batch_size, SEQ_LEN)), VOCAB).float()

    def empty_change(self, x):
        return torch.zeros(SEQ_LEN)

    def proposals(self, x):
        """Every substitution of one non-padding token by another token."""
        positions = (x != PAD).nonzero().flatten().tolist()
        candidates, changes = [], []
        for p in positions:
            for token in range(1, VOCAB):
                if token == int(x[p]):
                    continue
                candidate = x.clone()
                candidate[p] = token
                change = torch.zeros(SEQ_LEN)
                change[p] = 1.0
                candidates.append(candidate)
                changes.append(change)
        return torch.stack(candidates), torch.stack(changes)


class ToyProteinGenerator(_ToyDiscreteGenerator):
    """Latent = a continuous relaxation ``2x - 1`` of the one-hot residues."""

    def encode(self, x, t=1.0, stochastic=None, num_steps=None):
        return x * 2.0 - 1.0

    def decode(self, z, t=1.0, stochastic=None, num_steps=None):
        return F.one_hot(z.argmax(-2), N_AMINO).transpose(-1, -2).float()

    def sample_z(self, batch_size=1):
        return torch.randn(batch_size, N_AMINO, PROTEIN_LEN)

    def empty_change(self, x):
        return torch.zeros(PROTEIN_LEN)

    def proposals(self, x):
        """Every single-residue substitution (a point mutation)."""
        current = x.argmax(-2)
        candidates, changes = [], []
        for p in range(PROTEIN_LEN):
            for residue in range(N_AMINO):
                if residue == int(current[p]):
                    continue
                candidate = x.clone()
                candidate[:, p] = 0.0
                candidate[residue, p] = 1.0
                change = torch.zeros(PROTEIN_LEN)
                change[p] = 1.0
                candidates.append(candidate)
                changes.append(change)
        return torch.stack(candidates), torch.stack(changes)


# ---------------------------------------------------------------------------
# an explainer that produces the same record CounterfactualExplainer does
# ---------------------------------------------------------------------------


class ToyExplainerConfig(ExplainerConfig):
    config_name: str = "ToyExplainerConfig"
    explainer_type: str = "ToyExplainer"
    max_edits: int = 3
    target_confidence_goal: float = 0.6


class ToyExplainer(ExplainerInterface):
    """Explain a batch by asking the generator for a counterfactual of every sample.

    The dict it returns uses the key names of ``CounterfactualExplainer``'s
    batch record, so a teacher or an adaptor written against that record
    reads a graph, text or protein explanation the same way.
    """

    def __init__(
        self,
        config,
        device="cpu",
        predictor_dataset=None,
        predictor=None,
        generator=None,
        **kwargs,
    ):
        self.explainer_config = config
        self.device = device
        self.predictor_datasets = predictor_dataset
        self.predictor = predictor
        self.generator = generator

    @torch.no_grad()
    def explain_batch(self, batch, **args):
        x, y = batch
        probabilities = torch.softmax(self.predictor(x), -1)
        y_source = probabilities.argmax(-1)
        y_target = 1 - y_source  # binary tasks: the other class
        x_cf, z_diff, end_conf, x_list, _, success = self.generator.edit(
            x_in=x,
            target_confidence_goal=self.explainer_config.target_confidence_goal,
            source_classes=y_source,
            target_classes=y_target,
            predictor=self.predictor,
            explainer_config=self.explainer_config,
            predictor_datasets=self.predictor_datasets,
        )
        start_conf = probabilities[torch.arange(len(y_target)), y_target]
        return {
            "x_list": x_list,
            "y_list": [int(v) for v in y],
            "y_source_list": [int(v) for v in y_source],
            "y_target_list": [int(v) for v in y_target],
            "x_counterfactual_list": x_cf,
            "z_difference_list": z_diff,
            "y_target_start_confidence_list": [float(c) for c in start_conf],
            "y_target_end_confidence_list": [float(c) for c in end_conf],
            "success_list": success.tolist(),
        }


# ---------------------------------------------------------------------------
# registry tables and helpers the tests share
# ---------------------------------------------------------------------------

DOMAINS = {
    "graph": types.SimpleNamespace(
        name="graph",
        dataset=ToyGraphDataset,
        predictor=ToyGCN,
        generator=ToyGraphGenerator,
        rule=lambda x: has_triangle(adjacency(x)).long(),
    ),
    "text": types.SimpleNamespace(
        name="text",
        dataset=ToyTextDataset,
        predictor=ToyTextClassifier,
        generator=ToyTextGenerator,
        rule=lambda x: (keyword_count(x) >= 2).long(),
    ),
    "protein": types.SimpleNamespace(
        name="protein",
        dataset=ToyProteinDataset,
        predictor=ToyProteinCNN,
        generator=ToyProteinGenerator,
        rule=lambda x: (hydrophobic_fraction(x) > 0.4).long(),
    ),
}

#: what a plugin would hand to ``peal.registry.register`` (or an entry point)
REGISTRATIONS = {
    "datasets": {d.dataset.__name__: d.dataset for d in DOMAINS.values()},
    "generators": {d.generator.__name__: d.generator for d in DOMAINS.values()},
    "explainers": {"ToyExplainer": ToyExplainer},
    "configs": {
        "ToyGeneratorConfig": ToyGeneratorConfig,
        "ToyExplainerConfig": ToyExplainerConfig,
    },
}


def data_config(domain, **overrides):
    """A ``DataConfig`` selecting the domain's dataset class by name."""
    fields = dict(
        dataset_class=domain.dataset.__name__,
        input_type=domain.name,
        output_type="singleclass",
        input_size=list(domain.dataset.input_size),
        output_size=[2],
        num_samples=96,
        split=[0.75, 0.875],
        seed=0,
    )
    fields.update(overrides)
    return DataConfig(**fields)


def quick_fit(model, dataset, steps=200, lr=0.05, seed=0):
    """Full-batch Adam on the dataset's tensors; enough for the toy rules."""
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x, y = dataset.x, dataset.y
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        F.cross_entropy(model(x), y).backward()
        optimizer.step()
    return model.eval()
