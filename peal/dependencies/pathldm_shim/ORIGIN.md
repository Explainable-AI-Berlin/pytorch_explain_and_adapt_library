# pathldm_shim — PEAL's own code

`sitecustomize.py` (38 lines) is a PEAL-side shim that fixes `sys.path` so the
vendored PathLDM package can be imported without its original standalone
install. It is not third-party code.

Consider moving it under `peal/` proper, so that `peal/dependencies/` contains
only external code.
