"""Tabular datasets for the symbolic (non-image) CFKD/DiDAE experiments.

Each class wraps ``SymbolicDataset`` around one ``data.csv`` with a ``Target``
column and numeric-coded features. When the csv is missing the constructor
tries to build it: the synthetic Circle data is generated locally, Adult is
downloaded via the Kaggle API, COMPAS from the FairML GitHub mirror and German
Credit from OpenML. The constructors also set ``config.input_size`` from the
csv header so the tabular MLP predictors and the tabular DDPM know their width.
"""

import os

import pandas as pd

from peal.data.datasets import SymbolicDataset
from peal.log import get_logger

_log = get_logger(__name__)


def setup_kaggle_api():
    """Helper to ensure Kaggle API is authenticated, prompting the user if necessary.

    If ``~/.kaggle/kaggle.json`` does not exist the username and key are read
    from stdin and written there with mode ``0o600``.

    Returns
    -------
    kaggle.api.kaggle_api_extended.KaggleApi
        An authenticated API client.
    """
    import os
    import json

    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")

    if not os.path.exists(kaggle_json_path):
        os.makedirs(kaggle_dir, exist_ok=True)
        _log.info("%s", "Kaggle API credentials not found.")
        username = input("Please enter your Kaggle username: ").strip()
        key = input("Please enter your Kaggle API key: ").strip()

        with open(kaggle_json_path, "w") as f:
            json.dump({"username": username, "key": key}, f)
        os.chmod(kaggle_json_path, 0o600)
        _log.info("%s", f"Credentials saved to {kaggle_json_path}.")

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


from peal.data.dataset_generators import CircleDatasetGenerator


class CircleDataset(SymbolicDataset):
    """Synthetic 2-D "circle" dataset with a planted confounder.

    If ``config.dataset_path`` does not exist the data is generated first with
    ``CircleDatasetGenerator``; otherwise this is a plain ``SymbolicDataset``.

    Parameters
    ----------
    mode : str
        Split to load (``"train"``, ``"val"`` or ``"test"``).
    config : DataConfig
        Dataset configuration; ``dataset_path`` is where the csv lives or is
        generated.
    **kwargs
        Forwarded to ``SymbolicDataset``.
    """

    def __init__(self, mode, config, **kwargs):
        """Generate the csv if needed and build the ``SymbolicDataset``."""
        if not os.path.exists(config.dataset_path):
            # use the circle dataset generator
            circle_dataset_generator = CircleDatasetGenerator(config)
            circle_dataset_generator.generate_dataset()
        super(CircleDataset, self).__init__(mode=mode, config=config, **kwargs)

    def calculate_outlier_score(self, x):
        """Outlier scores of a batch; tabular data has none, so all zeros.

        Parameters
        ----------
        x : torch.Tensor
            Batch of samples ``[B, ...]``.

        Returns
        -------
        dict
            ``{"absolute": zeros[B], "relative": zeros[B]}``; ``"relative"``
            is divided by ``reference_outlier_scores`` when that is set.
        """
        import torch

        outlier_scores = {"absolute": torch.zeros(x.shape[0], device=x.device)}
        if (
            hasattr(self, "reference_outlier_scores")
            and self.reference_outlier_scores is not None
        ):
            outlier_scores["relative"] = outlier_scores["absolute"] / (
                self.reference_outlier_scores + 1e-8
            )
        else:
            outlier_scores["relative"] = outlier_scores["absolute"]
        return outlier_scores


#: The published tabular Adult cell was produced by a one-off curation script
#: (``tools/curate_data_adult.py``) whose one-hot column names are what
#: ``configs/tabular_experiments/adaptors/adult_cfkd_dice.yaml`` selects, e.g.
#: ``occupation_Professional`` and ``gender_Male``. ``AdultDataset`` used to
#: category-code the table instead, producing integer ``occupation`` and
#: ``gender`` columns that no committed config can consume, so the cell could
#: not run from a clean clone. The curation now lives here, which is the only
#: arrangement where downloading and running both work with no manual step.
ADULT_FEATURES = [
    "hours-per-week",
    "educational-num",
    "occupation",
    "workclass",
    "race",
    "age",
    "marital-status",
    "gender",
]
#: the raw UCI table spells two of those differently from the Kaggle mirror
ADULT_RENAME = {"education-num": "educational-num", "sex": "gender", "income": "Target"}
ADULT_WORKCLASS = {
    "?": "Other/Unknown",
    "Federal-gov": "Government",
    "Local-gov": "Government",
    "State-gov": "Government",
    "Self-emp-inc": "Self-Employed",
    "Self-emp-not-inc": "Self-Employed",
    "Never-worked": "Other/Unknown",
    "Without-pay": "Other/Unknown",
    "Other": "Other/Unknown",
    "Unknown": "Other/Unknown",
}
ADULT_OCCUPATION = {
    "?": "Other/Unknown",
    "Adm-clerical": "White-Collar",
    "Craft-repair": "Blue-Collar",
    "Exec-managerial": "White-Collar",
    "Farming-fishing": "Blue-Collar",
    "Handlers-cleaners": "Blue-Collar",
    "Machine-op-inspct": "Blue-Collar",
    "Other-service": "Service",
    "Priv-house-serv": "Service",
    "Prof-specialty": "Professional",
    "Protective-serv": "Service",
    "Tech-support": "Service",
    "Transport-moving": "Blue-Collar",
    "Unknown": "Other/Unknown",
    "Armed-Forces": "Other/Unknown",
}
ADULT_MARITAL = {
    "Married-AF-spouse": "Married",
    "Married-civ-spouse": "Married",
    "Married-spouse-absent": "Married",
    "Never-married": "Single",
}


#: The copy of Adult on this cluster is not the raw table: it is the output of the
#: OLD AdultDataset path, which category-coded every string column, so the category
#: names the curation needs are gone. pandas assigns codes in sorted order, so they
#: can be mapped back exactly -- but only if the code range matches the category
#: list, which `decode_adult_codes` checks before trusting it.
ADULT_CATEGORIES = {
    "workclass": [
        "?",
        "Federal-gov",
        "Local-gov",
        "Never-worked",
        "Private",
        "Self-emp-inc",
        "Self-emp-not-inc",
        "State-gov",
        "Without-pay",
    ],
    "occupation": [
        "?",
        "Adm-clerical",
        "Armed-Forces",
        "Craft-repair",
        "Exec-managerial",
        "Farming-fishing",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Other-service",
        "Priv-house-serv",
        "Prof-specialty",
        "Protective-serv",
        "Sales",
        "Tech-support",
        "Transport-moving",
    ],
    "marital-status": [
        "Divorced",
        "Married-AF-spouse",
        "Married-civ-spouse",
        "Married-spouse-absent",
        "Never-married",
        "Separated",
        "Widowed",
    ],
    "race": [
        "Amer-Indian-Eskimo",
        "Asian-Pac-Islander",
        "Black",
        "Other",
        "White",
    ],
    "gender": ["Female", "Male"],
    "sex": ["Female", "Male"],
}


def decode_adult_codes(df):
    """Map an integer-coded Adult table back to its category strings.

    Only used when the ``data.csv`` on disk is the category-coded artefact of the
    superseded code path rather than a raw download. Refuses rather than guesses:
    a column is decoded only when its observed codes are exactly ``0..len(cats)-1``.

    Parameters
    ----------
    df : pandas.DataFrame
        Table whose categorical columns hold integer codes.

    Returns
    -------
    pandas.DataFrame
        Copy with those columns replaced by their category strings.
    """
    df = df.copy()
    for col, cats in ADULT_CATEGORIES.items():
        if col not in df.columns or df[col].dtype == object:
            continue
        # The Kaggle csv writes missing values as "?", which pandas reads as NaN
        # and `cat.codes` then writes as -1, so "?" is not among the coded
        # categories: the real ones are the sorted remainder starting at 0.
        coded = [c for c in cats if c != "?"]
        observed = sorted(int(v) for v in df[col].dropna().unique())
        expected = ([-1] if -1 in observed else []) + list(range(len(coded)))
        if observed != expected:
            raise ValueError(
                f"cannot decode Adult column {col!r}: codes {observed} do not match "
                f"the expected {expected} for {len(coded)} categories, so the "
                "category order is unknown"
            )
        df[col] = df[col].map(lambda v: "?" if int(v) < 0 else coded[int(v)])
    return df


def curate_adult_frame(df):
    """Turn a raw Adult table into the one-hot encoding the configs select.

    Reproduces ``tools/curate_data_adult.py`` step for step, so the columns and
    the scaling match the ones behind the published number: keep eight features
    plus the target, binarise the target at ``>50K``, merge the sparse
    ``workclass``, ``occupation`` and ``marital-status`` categories, one-hot
    encode with ``drop_first``, and scale the numeric columns to [-1, 1].

    Accepts either spelling of the two columns the UCI and Kaggle versions
    disagree on (``education-num``/``educational-num``, ``sex``/``gender``).

    Parameters
    ----------
    df : pandas.DataFrame
        Raw Adult table with the original category strings.

    Returns
    -------
    pandas.DataFrame
        Curated table: one-hot feature columns plus ``Target``.
    """
    from sklearn.preprocessing import MinMaxScaler

    df = df.rename(columns={k: v for k, v in ADULT_RENAME.items() if k in df.columns})
    if any(
        df[c].dtype != object for c in ("workclass", "occupation") if c in df.columns
    ):
        df = decode_adult_codes(df)
    missing = [c for c in ADULT_FEATURES + ["Target"] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Adult table is missing {missing}; got columns {list(df.columns)}"
        )
    df = df[ADULT_FEATURES + ["Target"]].copy()
    # The raw table spells the target ">50K"/"<=50K"; a table that has already
    # been through the superseded code path carries it as 0/1, and mapping that
    # through the string test would silently zero every label.
    if df["Target"].dtype == object:
        df["Target"] = df["Target"].apply(
            lambda x: 1 if str(x).strip().startswith(">50K") else 0
        )
    else:
        df["Target"] = (df["Target"] > 0).astype(int)
    df["workclass"] = df["workclass"].replace(ADULT_WORKCLASS).astype("object")
    df["occupation"] = df["occupation"].replace(ADULT_OCCUPATION).astype("object")
    df["marital-status"] = df["marital-status"].replace(ADULT_MARITAL).astype("object")

    object_columns = df.select_dtypes(include=["object"]).columns
    df = pd.get_dummies(df, columns=object_columns, drop_first=True)
    numeric = df.select_dtypes(include=["int64", "float64"]).columns.drop("Target")
    df[numeric] = MinMaxScaler(feature_range=(-1, 1)).fit_transform(df[numeric])
    # Target LAST. `pd.get_dummies` appends the one-hot columns after the ones it did
    # not touch, which leaves Target in the middle -- and `SymbolicDataset` falls back
    # to `x = data[:-1]` when a predictor config gives no `x_selection`, so a
    # middle Target is fed to the model as a feature. The committed Adult predictor
    # config has no x_selection, and with Target in the middle it trained to 100 per
    # cent test accuracy on a dataset whose state of the art is about 87: pure label
    # leakage. Ordering the target last makes that fallback correct.
    columns = [c for c in df.columns if c != "Target"] + ["Target"]
    return df[columns].astype(float)


class AdultDataset(SymbolicDataset):
    """UCI Adult income dataset (``wenruliu/adult-income-dataset`` on Kaggle).

    On first use the csv is downloaded with the Kaggle API and passed through
    :func:`curate_adult_frame`, which produces the one-hot encoding the committed
    configs select; the raw table is kept as ``data_original.csv`` and the result
    written to ``<dataset_path>/data.csv``. An existing ``data.csv`` in the older
    category-coded encoding is curated in place on first use, because no config
    can consume that encoding. ``config.input_size`` is then set from the header.

    Parameters
    ----------
    mode : str
        Split to load.
    config : DataConfig
        Dataset configuration; ``dataset_path`` is the directory of ``data.csv``
        and ``input_size`` is overwritten.
    **kwargs
        Forwarded to ``SymbolicDataset``.
    """

    def __init__(self, mode, config, **kwargs):
        """Download/convert ``data.csv`` if missing, then build the dataset."""
        if not os.path.exists(f"{config.dataset_path}/data.csv"):
            _log.info(
                "%s",
                f"Dataset path {config.dataset_path} not found. Attempting to download via Kaggle API...",
            )
            os.makedirs(config.dataset_path, exist_ok=True)
            try:
                api = setup_kaggle_api()
                api.dataset_download_files(
                    "wenruliu/adult-income-dataset",
                    path=config.dataset_path,
                    unzip=True,
                )
                # Ensure the downloaded file is processed for SymbolicDataset conformity
                csv_file = f"{config.dataset_path}/adult.csv"
                if os.path.exists(csv_file):
                    # Keep the raw table: the curation is lossy and the script
                    # this reproduces read `data_original.csv`.
                    os.replace(csv_file, f"{config.dataset_path}/data_original.csv")
                    curate_adult_frame(
                        pd.read_csv(f"{config.dataset_path}/data_original.csv")
                    ).to_csv(f"{config.dataset_path}/data.csv", index=False)
            except Exception as e:
                _log.info("%s", f"Failed to download Adult dataset automatically: {e}")
                _log.info(
                    "%s",
                    f"Please manually download the Adult dataset from Kaggle and place it as: {config.dataset_path}/data.csv",
                )
                import time

                time.sleep(5)

        # A `data.csv` that is already there may predate the curation -- the copy
        # shipped on this cluster is the raw UCI table. Curate it in place rather
        # than failing later with `'educational-num' is not in list`, and keep the
        # original beside it.
        header = open(f"{config.dataset_path}/data.csv").readline()
        if "educational-num" not in header:
            _log.info(
                "%s",
                f"{config.dataset_path}/data.csv is not the curated encoding the "
                "configs select; curating it and keeping the original as "
                "data_original.csv",
            )
            raw = f"{config.dataset_path}/data_original.csv"
            if not os.path.exists(raw):
                os.replace(f"{config.dataset_path}/data.csv", raw)
            curate_adult_frame(pd.read_csv(raw)).to_csv(
                f"{config.dataset_path}/data.csv", index=False
            )

        with open(f"{config.dataset_path}/data.csv", "r") as f:
            config.input_size = [len(f.readline().strip().split(",")) - 1]
        super(AdultDataset, self).__init__(mode, config, **kwargs)

    def calculate_outlier_score(self, x):
        """Outlier scores of a batch; tabular data has none, so all zeros.

        Parameters
        ----------
        x : torch.Tensor
            Batch of samples ``[B, ...]``.

        Returns
        -------
        dict
            ``{"absolute": zeros[B], "relative": zeros[B]}``.
        """
        import torch

        outlier_scores = {"absolute": torch.zeros(x.shape[0], device=x.device)}
        if (
            hasattr(self, "reference_outlier_scores")
            and self.reference_outlier_scores is not None
        ):
            outlier_scores["relative"] = outlier_scores["absolute"] / (
                self.reference_outlier_scores + 1e-8
            )
        else:
            outlier_scores["relative"] = outlier_scores["absolute"]
        return outlier_scores


class CompassDataset(SymbolicDataset):
    """ProPublica COMPAS recidivism dataset (FairML preprocessed csv).

    On first use the csv is fetched from the DataResponsibly/fairDAGs GitHub
    mirror, ``Two_yr_Recidivism`` becomes ``Target``, ``African_American`` is
    copied to an ``is_black`` column (the confounder the teachers look at),
    categorical columns are integer-coded and everything is cast to float
    before writing ``<dataset_path>/data.csv``. ``config.input_size`` is set
    from the csv header.

    Parameters
    ----------
    mode : str
        Split to load.
    config : DataConfig
        Dataset configuration; ``dataset_path`` is the directory of ``data.csv``.
    **kwargs
        Forwarded to ``SymbolicDataset``.
    """

    def __init__(self, mode, config, **kwargs):
        """Download/convert ``data.csv`` if missing, then build the dataset."""
        if not os.path.exists(f"{config.dataset_path}/data.csv"):
            _log.info(
                "%s",
                f"Dataset path {config.dataset_path} not found. Attempting to download COMPASS via Kaggle API...",
            )
            os.makedirs(config.dataset_path, exist_ok=True)
            try:
                _log.info("%s", "Attempting to fetch COMPASS from FairML GitHub...")
                url = "https://raw.githubusercontent.com/DataResponsibly/fairDAGs/master/data/compas/propublica_data_for_fairml.csv"
                df = pd.read_csv(url)
                if "Two_yr_Recidivism" in df.columns:
                    df["Target"] = df.pop("Two_yr_Recidivism").astype(float)
                # Ensure is_black is available for the teacher
                if "African_American" in df.columns:
                    df["is_black"] = df["African_American"].astype(float)

                for col in df.columns:
                    if df[col].dtype == "object" or df[col].dtype.name == "category":
                        df[col] = df[col].astype("category").cat.codes
                df = df.astype(float)
                df.to_csv(f"{config.dataset_path}/data.csv", index=False)
            except Exception as e:
                _log.info(
                    "%s", f"Failed to download COMPASS dataset automatically: {e}"
                )
                _log.info(
                    "%s",
                    f"Please manually place the COMPASS dataset as: {config.dataset_path}/data.csv",
                )

        with open(f"{config.dataset_path}/data.csv", "r") as f:
            config.input_size = [len(f.readline().strip().split(",")) - 1]
        super(CompassDataset, self).__init__(mode, config, **kwargs)

    def calculate_outlier_score(self, x):
        """Outlier scores of a batch; tabular data has none, so all zeros.

        Parameters
        ----------
        x : torch.Tensor
            Batch of samples ``[B, ...]``.

        Returns
        -------
        dict
            ``{"absolute": zeros[B], "relative": zeros[B]}``.
        """
        import torch

        outlier_scores = {"absolute": torch.zeros(x.shape[0], device=x.device)}
        if (
            hasattr(self, "reference_outlier_scores")
            and self.reference_outlier_scores is not None
        ):
            outlier_scores["relative"] = outlier_scores["absolute"] / (
                self.reference_outlier_scores + 1e-8
            )
        else:
            outlier_scores["relative"] = outlier_scores["absolute"]
        return outlier_scores


class GermanDataset(SymbolicDataset):
    """German Credit dataset (OpenML ``credit-g``).

    On first use the frame is fetched with ``sklearn.datasets.fetch_openml``,
    ``class`` is mapped to ``Target`` (good=1, bad=0), a binary ``sex`` column
    is derived from ``personal_status`` as the confounder, categorical columns
    are integer-coded and everything is cast to float before writing
    ``<dataset_path>/data.csv``. ``config.input_size`` is set from the header.

    Parameters
    ----------
    mode : str
        Split to load.
    config : DataConfig
        Dataset configuration; ``dataset_path`` is the directory of ``data.csv``.
    **kwargs
        Forwarded to ``SymbolicDataset``.
    """

    def __init__(self, mode, config, **kwargs):
        """Download/convert ``data.csv`` if missing, then build the dataset."""
        if not os.path.exists(f"{config.dataset_path}/data.csv"):
            _log.info(
                "%s",
                f"Dataset path {config.dataset_path} not found. Attempting to download German Credit via Kaggle API...",
            )
            os.makedirs(config.dataset_path, exist_ok=True)
            try:
                from sklearn.datasets import fetch_openml

                _log.info("%s", "Attempting to fetch German Credit from OpenML...")
                data = fetch_openml(name="credit-g", version=1, as_frame=True)
                df = data.frame
                # Mapping target
                if "class" in df.columns:
                    df["Target"] = df["class"].map({"good": 1, "bad": 0}).astype(float)
                # Mapping confounder 'sex'
                if "personal_status" in df.columns:
                    df["sex"] = df["personal_status"].apply(
                        lambda x: 1.0 if "male" in str(x).lower() else 0.0
                    )

                for col in df.columns:
                    if df[col].dtype.name == "category" or df[col].dtype == "object":
                        df[col] = df[col].astype("category").cat.codes
                df = df.astype(float)
                df.to_csv(f"{config.dataset_path}/data.csv", index=False)
            except Exception as e:
                _log.info("%s", f"Failed to download German dataset automatically: {e}")
                _log.info(
                    "%s",
                    f"Please manually place the German dataset as: {config.dataset_path}/data.csv",
                )
                import time

                time.sleep(5)

        with open(f"{config.dataset_path}/data.csv", "r") as f:
            config.input_size = [len(f.readline().strip().split(",")) - 1]
        super(GermanDataset, self).__init__(mode, config, **kwargs)

    def calculate_outlier_score(self, x):
        """Outlier scores of a batch; tabular data has none, so all zeros.

        Parameters
        ----------
        x : torch.Tensor
            Batch of samples ``[B, ...]``.

        Returns
        -------
        dict
            ``{"absolute": zeros[B], "relative": zeros[B]}``.
        """
        import torch

        outlier_scores = {"absolute": torch.zeros(x.shape[0], device=x.device)}
        if (
            hasattr(self, "reference_outlier_scores")
            and self.reference_outlier_scores is not None
        ):
            outlier_scores["relative"] = outlier_scores["absolute"] / (
                self.reference_outlier_scores + 1e-8
            )
        else:
            outlier_scores["relative"] = outlier_scores["absolute"]
        return outlier_scores
