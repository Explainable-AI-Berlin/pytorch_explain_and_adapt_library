"""Build the DiDAE paper's Table 1 (LaTeX) from TensorBoard logs of CFKD runs.

For each dataset (Square, CelebA-Blond, Camelyon) and each explainer (DAE,
DiME, ACE, FastDiME, SCE, DiDAE) the adaptor config listed in ``configs`` is
opened, its ``base_dir`` is resolved against ``$PEAL_RUNS`` and the scalars of
``<base_dir>/logs`` are read. Run as a script it prints one table per seed
directory (``PEAL_RUNS``, ``PEAL_RUNS1..3``) plus a mean/std table over the
three seeds. Cells read from step 1 instead of step 0 are set in italics.
"""

import os
import yaml
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from peal.log import get_logger

_log = get_logger(__name__)


datasets = ["Square", "CelebA-Blond", "Camelyon"]
methods = ["DAE", "DiME", "ACE", "FastDiME", "SCE", "DiDAE (ours)"]

configs = {
    "Square": {
        "DAE": "configs/didae_experiments/adaptors/square1k_resnet18_poisoned098_dae_original_cfkd.yaml",
        "DiME": "configs/didae_experiments/adaptors/square1k_resnet18_poisoned098_dime_cfkd.yaml",
        "ACE": "configs/didae_experiments/adaptors/square1k_resnet18_poisoned098_ace_cfkd.yaml",
        "FastDiME": "configs/didae_experiments/adaptors/square1k_resnet18_poisoned098_fastdime_cfkd.yaml",
        "SCE": "configs/didae_experiments/adaptors/square1k_resnet18_poisoned098_sce_cfkd.yaml",
        "DiDAE (ours)": "configs/didae_experiments/adaptors/square1k_resnet18_poisoned098_didae_procrustes_cfkd.yaml",
    },
    "CelebA-Blond": {
        "DAE": "configs/didae_experiments/adaptors/celeba1kx098_resnet18_dae_original_cfkd.yaml",
        "DiME": "configs/didae_experiments/adaptors/celeba1kx098_resnet18_dime_cfkd.yaml",
        "ACE": "configs/didae_experiments/adaptors/celeba1kx098_resnet18_ace_cfkd.yaml",
        "FastDiME": "configs/didae_experiments/adaptors/celeba1kx098_resnet18_fastdime_cfkd.yaml",
        "SCE": "configs/didae_experiments/adaptors/celeba1kx098_resnet18_sce_cfkd.yaml",
        "DiDAE (ours)": "configs/didae_experiments/adaptors/celeba1kx098_resnet18_didae_openclip_cfkd.yaml",
    },
    "Camelyon": {
        "DAE": "configs/didae_experiments/adaptors/camelyon_1k_resnet18_poisoning098_dae_original_cfkd.yaml",
        "DiME": "configs/didae_experiments/adaptors/camelyon17_1k_poisoned098_dime_cfkd.yaml",
        "ACE": "configs/didae_experiments/adaptors/camelyon17_1k_poisoned098_ace_cfkd.yaml",
        "FastDiME": "configs/didae_experiments/adaptors/camelyon17_1k_poisoned098_fastdime_cfkd.yaml",
        "SCE": "configs/didae_experiments/adaptors/camelyon17_1k_poisoned098_sce_cfkd.yaml",
        "DiDAE (ours)": "configs/didae_experiments/adaptors/camelyon_1k_resnet18_poisoning098_didae_cfkd.yaml",
    },
}

metrics = {
    "NAFR": "validation_valid_counterfactual_rate",
    "Diversity": "validation_latent_diversity",
    "Sparsity": "validation_latent_sparsity",
    "NA": "test_accuracy",
    "Unbiasedness": "test_worst_group_accuracy",
    "CF/s": "validation_counterfactuals_per_second",
    "Gain": "gain",
}


def get_scalar(ea, tag, step):
    """Read one scalar from an event accumulator, falling back to ``step + 1``.

    Parameters
    ----------
    ea : EventAccumulator
        Loaded accumulator of a run's ``logs`` directory.
    tag : str
        Scalar tag, e.g. ``"validation_latent_diversity"``.
    step : int
        Preferred step.

    Returns
    -------
    tuple
        ``(value, from_fallback)``: the value at ``step``, or the value at
        ``step + 1`` with ``from_fallback=True``, or ``(nan, False)`` when the
        tag or both steps are missing.
    """
    try:
        items = ea.scalars.Items(tag)
        primary_val = float("nan")
        fallback_val = float("nan")
        for item in items:
            if item.step == step:
                primary_val = item.value
            elif item.step == step + 1:
                fallback_val = item.value
        if not np.isnan(primary_val):
            return primary_val, False
        if not np.isnan(fallback_val):
            return fallback_val, True
        return float("nan"), False
    except KeyError:
        return float("nan"), False


def create_table(runs_var_name, data_dict=None):
    """Collect the table cells for one seed directory.

    Parameters
    ----------
    runs_var_name : str
        Environment variable naming the runs directory (``"PEAL_RUNS"`` or
        ``"PEAL_RUNS<n>"``). If it is unset but ``PEAL_RUNS`` is, the
        directory ``$PEAL_RUNS<n>`` is used for ``PEAL_RUNS1..3``.
    data_dict : dict, optional
        Dictionary to fill; a new one is created when ``None``.

    Returns
    -------
    dict
        ``data_dict[dataset][method][column] = (value, from_fallback)`` for the
        columns in ``metrics``. Values are percentages except ``CF/s``.
        Missing configs, ``base_dir`` entries or log directories leave the
        ``nan`` placeholders in place.

    Notes
    -----
    The tags actually read differ from the ``metrics`` mapping: NAFR comes
    from ``validation_flip_rate_distilled``, NA from
    ``validation_non_adversarial_rate``, Unbiasedness from
    ``validation_unbiasedness`` (all at step 0) and Gain from ``gain`` at
    step 1. ``metrics`` only supplies the column names.
    """
    if data_dict is None:
        data_dict = {}

    runs_dir = os.environ.get(runs_var_name, "")
    if not runs_dir and os.environ.get("PEAL_RUNS", ""):
        if runs_var_name == "PEAL_RUNS1":
            runs_dir = os.environ.get("PEAL_RUNS", "") + "1"
        elif runs_var_name == "PEAL_RUNS2":
            runs_dir = os.environ.get("PEAL_RUNS", "") + "2"
        elif runs_var_name == "PEAL_RUNS3":
            runs_dir = os.environ.get("PEAL_RUNS", "") + "3"

    if not runs_dir:
        _log.info("%s", f"% Warning: env var {runs_var_name} not set.")

    for dataset in datasets:
        data_dict[dataset] = {}
        for method in methods:
            data_dict[dataset][method] = {m: (float("nan"), False) for m in metrics}
            config_path = configs[dataset][method]
            if not os.path.exists(config_path):
                continue
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            base_dir = config.get("base_dir", "")
            if not base_dir:
                continue

            # Resolve PEAL_RUNS
            base_dir = base_dir.replace("$PEAL_RUNS", runs_dir).replace(
                "${PEAL_RUNS}", runs_dir
            )
            log_dir = os.path.join(base_dir, "logs")

            if os.path.exists(log_dir):
                ea = EventAccumulator(log_dir)
                ea.Reload()

                nafr, nafr_it = get_scalar(ea, "validation_flip_rate_distilled", 0)
                data_dict[dataset][method]["NAFR"] = (nafr * 100, nafr_it)
                div, div_it = get_scalar(ea, "validation_latent_diversity", 0)
                data_dict[dataset][method]["Diversity"] = (div * 100, div_it)
                spa, spa_it = get_scalar(ea, "validation_latent_sparsity", 0)
                data_dict[dataset][method]["Sparsity"] = (spa * 100, spa_it)
                na, na_it = get_scalar(ea, "validation_non_adversarial_rate", 0)
                data_dict[dataset][method]["NA"] = (na * 100, na_it)

                unb, unb_it = get_scalar(ea, "validation_unbiasedness", 0)
                data_dict[dataset][method]["Unbiasedness"] = (unb * 100, unb_it)

                cfs, cfs_it = get_scalar(ea, "validation_counterfactuals_per_second", 0)
                data_dict[dataset][method]["CF/s"] = (cfs, cfs_it)

                gain, gain_it = get_scalar(ea, "gain", 1)
                data_dict[dataset][method]["Gain"] = (gain * 100, gain_it)
    return data_dict


def format_table(data_dict, title=""):
    """Render a :func:`create_table` result as a LaTeX ``table*`` environment.

    Parameters
    ----------
    data_dict : dict
        Output of :func:`create_table`.
    title : str
        Used for a leading LaTeX comment and the caption when non-empty.

    Returns
    -------
    str
        LaTeX source. ``nan`` cells become ``----``, fallback cells are
        wrapped in ``\textit{}``, ``CF/s`` is printed with two decimals and
        the DiDAE row is shaded.
    """
    latex = []
    if title:
        latex.append(f"% Table for {title}")
    latex.append(r"\begin{table*}[t!]")
    if title:
        latex.append(f"\\caption{{Results for {title}}}")
    latex.append(r"\centering \footnotesize")
    latex.append(r"\renewcommand{\arraystretch}{1.1}")
    latex.append(r"\setlength{\tabcolsep}{4pt}")
    latex.append(r"\begin{tabular}{llrcccccccc}")
    latex.append(r"\toprule")
    latex.append(
        r" & & & \multicolumn{6}{c}{\cellcolor{CadetBlue!20}\textbf{desiderata}}\\"
    )
    latex.append(
        r" & & & \multicolumn{2}{c|}{\cellcolor{CadetBlue!20}{sufficiency}} & \multicolumn{1}{c|}{\cellcolor{CadetBlue!20}{understandability}} & \multicolumn{2}{c|}{\cellcolor{CadetBlue!20}{fidelity}} & \multicolumn{1}{c|}{\cellcolor{CadetBlue!20}{efficiency}}\\"
    )
    latex.append(
        r"Dataset & Method & & \cellcolor{CadetBlue!20}(NAFR) & \multicolumn{1}{c|}{\cellcolor{CadetBlue!20}(Diversity)} & \multicolumn{1}{c|}{\cellcolor{CadetBlue!20}(Sparsity)} & \cellcolor{CadetBlue!20}(NA) & \multicolumn{1}{c|}{\cellcolor{CadetBlue!20}(Unbiasedness)} & \multicolumn{1}{c|}{\cellcolor{CadetBlue!20}(CF/s)} & Gain \\"
    )

    for i, dataset in enumerate(datasets):
        for j, method in enumerate(methods):
            row_prefix = r"\multirow{6}{*}{" + dataset + "}" if j == 0 else ""
            if method == "DiDAE (ours)":
                row_prefix = r"\rowcolor{gray!10}\cellcolor{white} " + row_prefix

            vals = data_dict[dataset][method]

            # Format values
            # Format values
            def fmt(val_tuple, fmt_str="{:.1f}"):
                val, is_it = val_tuple
                if np.isnan(val):
                    return "----"
                s = fmt_str.format(val)
                return f"\\textit{{{s}}}" if is_it else s

            nafr = fmt(vals["NAFR"])
            div = fmt(vals["Diversity"])
            spa = fmt(vals["Sparsity"])
            na = fmt(vals["NA"])
            unb = fmt(vals["Unbiasedness"])
            cfs = fmt(vals["CF/s"], "{:.2f}")
            gain = fmt(vals["Gain"])

            row = f"{row_prefix:<35} & {method:<12} & & {nafr:>4} & {div:>4} & {spa:>4} & {na:>4} & {unb:>4} & $\\sim$ {cfs:>5} & {gain:>4} \\\\"
            latex.append(row)
        if i < len(datasets) - 1:
            latex.append(r"\midrule")
            latex.append("")

    latex.append(r"\midrule")
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\label{tab:quantitative_results_main}")
    latex.append(r"\end{table*}")
    latex.append("\n")
    return "\n".join(latex)


if __name__ == "__main__":
    runs_vars = ["PEAL_RUNS1", "PEAL_RUNS2", "PEAL_RUNS3"]
    all_data = []

    # Process default runs but don't append to all_data
    default_data = create_table("PEAL_RUNS")
    _log.info("%s", format_table(default_data, title="PEAL_RUNS (Default)"))

    for var in runs_vars:
        data = create_table(var)
        all_data.append(data)
        _log.info("%s", format_table(data, title=var))

    # Aggregate (mean and std)
    latex = []
    latex.append("% Table for AGGREGATED")
    latex.append(r"\begin{table*}[t!]")
    latex.append(r"\caption{Aggregated Results (Seeds 1, 2, 3)}")
    latex.append(r"\centering \footnotesize")
    latex.append(r"\renewcommand{\arraystretch}{1.1}")
    latex.append(r"\setlength{\tabcolsep}{4pt}")
    latex.append(r"\begin{tabular}{llrcccccccc}")
    latex.append(r"\toprule")
    latex.append(
        r" & & & \multicolumn{6}{c}{\cellcolor{CadetBlue!20}\textbf{desiderata}}\\"
    )
    latex.append(
        r" & & & \multicolumn{2}{c|}{\cellcolor{CadetBlue!20}{sufficiency}} & \multicolumn{1}{c|}{\cellcolor{CadetBlue!20}{understandability}} & \multicolumn{2}{c|}{\cellcolor{CadetBlue!20}{fidelity}} & \multicolumn{1}{c|}{\cellcolor{CadetBlue!20}{efficiency}}\\"
    )
    latex.append(
        r"Dataset & Method & & \cellcolor{CadetBlue!20}(NAFR) & \multicolumn{1}{c|}{\cellcolor{CadetBlue!20}(Diversity)} & \multicolumn{1}{c|}{\cellcolor{CadetBlue!20}(Sparsity)} & \cellcolor{CadetBlue!20}(NA) & \multicolumn{1}{c|}{\cellcolor{CadetBlue!20}(Unbiasedness)} & \multicolumn{1}{c|}{\cellcolor{CadetBlue!20}(CF/s)} & Gain \\"
    )

    for i, dataset in enumerate(datasets):
        for j, method in enumerate(methods):
            row_prefix = r"\multirow{6}{*}{" + dataset + "}" if j == 0 else ""
            if method == "DiDAE (ours)":
                row_prefix = r"\rowcolor{gray!10}\cellcolor{white} " + row_prefix

            def get_mean_std(m):
                vals = [
                    d[dataset][method][m][0]
                    for d in all_data
                    if not np.isnan(d[dataset][method][m][0])
                ]
                is_it = any(
                    d[dataset][method][m][1]
                    for d in all_data
                    if not np.isnan(d[dataset][method][m][0])
                )
                if vals:
                    s = f"{np.mean(vals):.1f}$\\pm${np.std(vals):.1f}"
                    return f"\\textit{{{s}}}" if is_it else s
                return "----"

            def get_mean_std_cfs(m):
                vals = [
                    d[dataset][method][m][0]
                    for d in all_data
                    if not np.isnan(d[dataset][method][m][0])
                ]
                is_it = any(
                    d[dataset][method][m][1]
                    for d in all_data
                    if not np.isnan(d[dataset][method][m][0])
                )
                if vals:
                    s = f"{np.mean(vals):.2f}$\\pm${np.std(vals):.2f}"
                    return f"\\textit{{{s}}}" if is_it else s
                return "----"

            nafr = get_mean_std("NAFR")
            div = get_mean_std("Diversity")
            spa = get_mean_std("Sparsity")
            na = get_mean_std("NA")
            unb = get_mean_std("Unbiasedness")
            cfs = get_mean_std_cfs("CF/s")
            gain = get_mean_std("Gain")

            row = f"{row_prefix:<35} & {method:<12} & & {nafr:>4} & {div:>4} & {spa:>4} & {na:>4} & {unb:>4} & $\\sim$ {cfs:>5} & {gain:>4} \\\\"
            latex.append(row)
        if i < len(datasets) - 1:
            latex.append(r"\midrule")
            latex.append("")

    latex.append(r"\midrule")
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\label{tab:quantitative_results_main_agg}")
    latex.append(r"\end{table*}")

    _log.info("%s", "\n".join(latex))
