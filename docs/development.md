# Development

## Formatting

All first-party Python is formatted with [black](https://black.readthedocs.io)
at 88 columns; the options live in `pyproject.toml`. The vendored third-party
research code under `peal/dependencies/` and the optional `external/` clone
are excluded so that upstream diffs stay readable.

```bash
pip install black==24.8.0
black .            # formats everything the config does not exclude
black --check .    # what CI and the hook verify
```

## Pre-commit hooks

`.pre-commit-config.yaml` runs black, a syntax check, a merge-marker check,
the config preflight and the no-debugger check on every commit. Install the
hooks once per clone:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files   # optional: check the whole tree now
```

A commit that touches unformatted Python fails and leaves the reformatted
files in the working tree; `git add` them and commit again.

## Docstrings

New code is documented in the NumPy docstring style so that Sphinx
(`napoleon`) renders it:

```python
def edit(self, x_in, target_confidence_goal, target_classes, classifier):
    """Move ``x_in`` across the decision boundary of ``classifier``.

    Parameters
    ----------
    x_in : torch.Tensor
        Batch of inputs in the dataset's processed format, shape (B, C, H, W).
    target_confidence_goal : float
        Confidence the counterfactual must reach for ``target_classes``.
    target_classes : torch.Tensor
        Target class per sample, shape (B,).
    classifier : torch.nn.Module
        The predictor being explained.

    Returns
    -------
    torch.Tensor
        The counterfactual batch, same shape as ``x_in``.
    """
```

## Building the documentation

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html`. The API reference under
`docs/reference/generated/` is regenerated on every build from the docstrings
and is not committed. Any third-party import that is not installed on the
build machine is mocked by `docs/conf.py`, so the site builds on a laptop
without torch; on a full environment the real modules are imported and the
type hints resolve.

## Publishing on GitHub Pages

`.github/workflows/docs.yml` builds the site on every push and deploys it to
GitHub Pages on pushes to `master` (and on manual dispatch). To enable it once:
in the repository settings under *Pages*, set the source to *GitHub Actions*.
The site then appears at `https://<owner>.github.io/<repository>/`.
