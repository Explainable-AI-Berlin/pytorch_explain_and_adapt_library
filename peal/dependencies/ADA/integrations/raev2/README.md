# RAEv2 integration

The ADA generator experiments build on upstream RAEv2 at commit:

```text
8a0d238f8dc3b261aba98b217f6c79c0182e8e94
```

The integration has two explicit parts:

- `patches/raev2-ada-development.patch` records modifications to upstream files;
- `overlay/` contains newly added cache, sampler, configuration, test, and
  diagnostic files.

Apply both with:

```bash
bash integrations/raev2/apply_overlay.sh
```

The current patch is a faithful snapshot of the development stack used by the
running experiments. It includes compatibility work broader than the final ADA
method and should be split into focused commits before a public release. It is
kept as a patch rather than silently vendored so reviewers can inspect exactly
what differs from upstream.

RAEv2 is licensed under CC BY-NC 4.0. The upstream license in the submodule
governs the RAEv2 source and derivative modifications in this integration.
