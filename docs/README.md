<!-- Not part of the Sphinx build; conf.py excludes it. -->
# PEAL documentation

The Sphinx sources of the PEAL documentation site.

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html      # or: cd docs && make html
```

Open `docs/_build/html/index.html`.

| Path | What it is |
| --- | --- |
| `index.rst` | site root and table of contents |
| `installation.md`, `architecture.md`, `usage.md`, `development.md` | the user guide; `usage.md` includes the repository README |
| `reference/index.rst` | the API reference root; the per-object pages under `reference/generated/` are regenerated on every build and are not committed |
| `_templates/autosummary/` | the module and class page templates |
| `conf.py` | build configuration, including the import mocking |

Any third-party import that is missing on the build machine is mocked by
`conf.py`, so the site builds without torch or a GPU. Nothing else needs to be
installed.

`.github/workflows/docs.yml` runs this build on every push and deploys it to
GitHub Pages from `master`. Enable it once under *Settings -> Pages* by setting
the source to *GitHub Actions*.

The other folders under `docs/` are not part of the site: `agent_protocols/`
holds the dual-agent research workflow, `paper_figures/` the figure scripts of
the DiDAE paper and `model_cards/` the model card of the published RAE weights.
