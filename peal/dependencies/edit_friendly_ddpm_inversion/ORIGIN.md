# edit_friendly_ddpm_inversion — PEAL's own implementation, currently unused

A from-the-paper implementation of edit-friendly DDPM inversion
(Huberman-Spiegelglas, Kulikov & Michaeli, CVPR 2024) adapted to a diffusion
autoencoder whose noise predictor is additionally conditioned on a semantic
code. It is PEAL's own code, not a fork.

**Nothing in PEAL imports it.** The implementation actually used at runtime is
the vendored upstream fork in `peal/dependencies/ddpm_inversion/`. Unless it is
wired up deliberately, this folder should be deleted to avoid two
implementations of the same method.
