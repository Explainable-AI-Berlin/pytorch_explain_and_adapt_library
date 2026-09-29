# CelebA / OpenAI-CLIP diffusion autoencoder: DiDAE helper scripts (2026-09-11)

All run inside the project container (`apptainer exec ... python_container.sif python <script>`),
with PEAL_BASE / PEAL_RUNS / PEAL_DATA exported. One GPU process at a time on the 16 GiB nodes.

| script | what it does |
|---|---|
| `make_slim_generator.py` | writes `$PEAL_RUNS/celeba/diffusion_autoencoder_openai_clip_vit_l14_slim/` = the generator's `final.ckpt` minus optimizer state and the frozen CLIP `encoder.*` keys (0.65 GB instead of 4.7 GB). Identical weights; re-run whenever `final.ckpt` improves. |
| `fit_dictionary.py <sd.yaml> <name>` | fits a sparse-dictionary config in that generator's z_sem space and saves it under the (real) generator dir, e.g. `procrustes_sae_celeba_openai_clip.yaml OrthogonalProcrustesDictionary` (2 comps) or `procrustes_sae_celeba_40comps.yaml OrthogonalProcrustesDictionary40Comps`. |
| `compute_bounds.py <dict_dir>` | writes `c_min_and_maxes.txt` for a dictionary that no DiDAE run has touched yet (CFKD's "dynamic" linesearch needs it). |
| `make_cfkd_on_discovered.py <didae_run> <out.yaml> <base_dir_name>` | CFKD config on the directions a DiDAE run marked `false` = DiDAE step 9 as a separate process (two generators in one process do not fit 16 GiB). Picks the `_b{2,3}` explainer matching the run's component_bounds_scale. |
| `didae_grid.py all|v1,v2` | the parameter grid (discovery half only, shared probe); writes `grid_summary.md`. |
| `gen_check_and_fit_procrustes.py`, `edit_ddim_test.py` | the generator diagnostics (raw vs EMA reconstruction, DDPM- vs DDIM-inversion edit realisation on the Male direction). Outputs in `<generator>/diagnostics_20260911/`. |
