"""Label-file parsers that turn csv/json annotations into tensors and split keys.

These helpers back the image and tabular datasets in ``peal.data``: they read a
label table, convert every row into a float tensor, optionally subsample the
rows so that a target attribute and a confounder co-occur with a prescribed
probability (the controlled "Clever Hans" datasets), optionally overwrite the
confounder column with human "spray" labels, and finally return the keys that
belong to the requested ``train``/``val``/``test`` partition. Splitting is made
deterministic with ``random.seed(0)``.
"""

import json
import torch
import random
import numpy as np
from peal.log import get_logger

_log = get_logger(__name__)


def parse_json(data_dir, config, mode, set_negative_to_zero=True):
    """Parse a json label file of a confounder-controlled dataset.

    The json maps sample ids to records with ``values`` (feature list),
    ``target`` and ``has_confounder``. Each record becomes a tensor
    ``[*values, target]``.

    Parameters
    ----------
    data_dir : str
        Path of the json file.
    config : DataConfig
        Needs ``confounding_factors`` (length 2) and either
        ``confounder_probability`` or ``full_confounder_config``, plus
        ``num_samples`` and ``split``.
    mode : str
        ``"train"``, ``"val"``, ``"test"`` or anything else for all keys.
    set_negative_to_zero : bool, optional
        Clamp negative feature values to zero.

    Returns
    -------
    tuple or None
        ``(data, keys_out)`` from ``process_confounder_data_controlled``, where
        ``data`` maps string keys (the record index) to tensors. Returns
        ``None`` when the config does not describe a controlled confounder
        setup (no other branch is implemented).
    """
    with open(data_dir, "r") as f:
        raw_data = json.load(f)

    if (
        not config.confounding_factors is None
        and len(config.confounding_factors) == 2
        and not (
            config.confounder_probability is None
            and config.full_confounder_config is None
        )
    ):

        def extract_instances_tensor_confounder(idx, line):
            """Turn one json record into ``(key, tensor, target, confounder)``.

            The returned tensor is ``[*line["values"], line["target"]]``, so
            the target is appended as the last attribute column.
            """
            key = str(idx)
            instances_tensor = torch.tensor(line["values"])
            attribute = line["target"]
            confounder = line["has_confounder"]
            instances_tensor = torch.cat([instances_tensor, torch.tensor([attribute])])
            return key, instances_tensor, attribute, confounder

        return process_confounder_data_controlled(
            raw_data=raw_data.values(),
            config=config,
            mode=mode,
            extract_instances_tensor=extract_instances_tensor_confounder,
            set_negative_to_zero=set_negative_to_zero,
        )


def parse_csv(
    data_dir,
    config,
    mode,
    key_type="idx",
    set_negative_to_zero=True,
    delimiter=",",
    raw_data=None,
):
    """Parse a csv label table into per-sample tensors and the keys of one split.

    The first row is the header (a leading row with fewer than two fields, e.g.
    a sample count, is skipped). Every other row is converted to a float
    tensor; fields that are not valid floats become ``-1.0``. Three selection
    mechanisms follow, in order:

    1. If ``config.confounding_factors`` has two entries and a confounder
       probability or ``full_confounder_config`` is set, rows are subsampled
       by ``process_confounder_data_controlled``.
    2. Otherwise all rows are kept (rows with ``has_mask != 1`` are dropped
       when ``config.has_hints``) and split either by ``config.split``
       fractions after a seeded shuffle, or, if ``split`` has not exactly two
       entries, by a ``split`` column holding 0/1/2 for train/val/test.
    3. If ``config.spray_label_file`` is set, its ``SprayLabel`` column
       overwrites the confounder column; samples whose spray label is not 0
       or 1 are dropped, and with ``config.spray_groups_balanced`` the four
       (target, confounder) groups are truncated to the smallest group size.

    Parameters
    ----------
    data_dir : str
        Path of the csv file (ignored when ``raw_data`` is given).
    config : DataConfig
        Uses ``x_selection``, ``confounding_factors``,
        ``confounder_probability``, ``full_confounder_config``, ``has_hints``,
        ``split``, ``spray_label_file`` and ``spray_groups_balanced``.
    mode : str
        ``"train"``, ``"val"`` or ``"test"``; any other value keeps all keys.
    key_type : str, optional
        ``"idx"`` keys samples by row number; ``"name"`` keys them by the
        column named ``config.x_selection`` (or column 0) and drops that and
        all preceding columns from the tensor and from ``attributes``.
    set_negative_to_zero : bool, optional
        Clamp negative values to zero.
    delimiter : str, optional
        Field separator.
    raw_data : list of str, optional
        Already-read lines to use instead of opening ``data_dir``.

    Returns
    -------
    attributes : list of str
        Column names of the returned tensors.
    data : dict
        Mapping key -> 1-D float tensor with one entry per attribute.
    keys_out : list of str
        Keys belonging to ``mode``, shuffled with ``random.seed(0)``.
    """
    if raw_data is None:
        raw_data = open(data_dir, "r").read().split("\n")
    # in case there is e.g. the number of instances in the first line
    if len(raw_data[0].split(delimiter)) < 2:
        raw_data = raw_data[1:]

    attributes = raw_data[0].split(delimiter)

    if config.x_selection in attributes:
        key_idx = attributes.index(config.x_selection)

    else:
        key_idx = 0
    if key_type == "name":
        attributes = attributes[key_idx + 1 :]

    raw_data = raw_data[1:]
    while "" in raw_data:
        raw_data.remove("")

    def extract_instances_tensor(idx, line):
        """Convert one csv line into ``(key, float tensor)``.

        The key is the row number (``key_type == "idx"``) or the value of the
        key column (``key_type == "name"``), in which case the key column and
        everything before it is stripped from the tensor. Empty fields are
        dropped and non-numeric fields become ``-1.0``.
        """
        instance_attributes = line.split(delimiter)
        if key_type == "idx":
            key = str(idx)

        elif key_type == "name":
            key = instance_attributes[key_idx]
            instance_attributes = instance_attributes[key_idx + 1 :]

        while "" in instance_attributes:
            instance_attributes.remove("")

        ''''# TODO is this a good choice?
        for i in range(len(instance_attributes)):
            if instance_attributes[i] == "":
                instance_attributes[i] = "0.0"'''

        def is_valid_float(s):
            """Return ``True`` if ``s`` can be parsed as a float."""
            try:
                float(s)
                return True
            except ValueError:
                return False

        instance_attributes_int = list(
            map(
                lambda x: float(x) if is_valid_float(x) else -1.0,
                instance_attributes,
            )
        )
        instances_tensor = torch.tensor(instance_attributes_int)
        return key, instances_tensor

    if (
        not config.confounding_factors is None
        and len(config.confounding_factors) == 2
        and not (
            config.confounder_probability is None
            and config.full_confounder_config is None
        )
    ):

        def extract_instances_tensor_confounder(idx, line):
            """Parse one csv line and read off target and confounder.

            Wraps ``extract_instances_tensor`` and additionally returns the
            two integer columns named by ``config.confounding_factors``, which
            ``process_confounder_data_controlled`` uses to build the groups.
            """
            selection_idx1 = attributes.index(config.confounding_factors[0])
            selection_idx2 = attributes.index(config.confounding_factors[1])
            key, instances_tensor = extract_instances_tensor(idx, line)
            attribute = int(instances_tensor[selection_idx1])
            confounder = int(instances_tensor[selection_idx2])
            return key, instances_tensor, attribute, confounder

        data, keys_out = process_confounder_data_controlled(
            raw_data=raw_data,
            config=config,
            mode=mode,
            extract_instances_tensor=extract_instances_tensor_confounder,
            set_negative_to_zero=set_negative_to_zero,
        )

    else:
        data = {}
        n = [0, 0]
        for idx, line in enumerate(raw_data):
            key, instances_tensor = extract_instances_tensor(idx, line)
            if (
                config.has_hints
                and "has_mask" in attributes
                and not instances_tensor[attributes.index("has_mask")] == 1
            ):
                continue

            if set_negative_to_zero:
                data[key] = torch.maximum(
                    torch.zeros_like(instances_tensor),
                    instances_tensor,
                )

            else:
                data[key] = instances_tensor

        keys = list(data.keys())
        if len(config.split) == 2:
            keys.sort()
            random.seed(0)
            random.shuffle(keys)
            if mode == "train":
                keys_out = keys[: int(len(keys) * config.split[0])]

            elif mode == "val":
                keys_out = keys[
                    int(len(keys) * config.split[0]) : int(len(keys) * config.split[1])
                ]

            elif mode == "test":
                keys_out = keys[int(len(keys) * config.split[1]) :]

            else:
                keys_out = keys

        else:
            mode_to_int = {"train": 0, "val": 1, "test": 2}
            mode_idx = attributes.index("split")
            keys_out = list(
                filter(lambda key: data[key][mode_idx] == mode_to_int[mode], keys)
            )

    if config.spray_label_file is not None:
        spray_data = open(config.spray_label_file, "r").read().split("\n")
        # in case there is e.g. the number of instances in the first line
        if len(spray_data[0].split(delimiter)) < 2:
            spray_data = spray_data[1:]

        spray_label_idx = spray_data[0].split(delimiter).index("SprayLabel")
        if key_type == "name":
            spray_label_idx -= key_idx + 1
        true_feature_index = attributes.index(config.confounding_factors[0])
        confounder_index = attributes.index(config.confounding_factors[1])

        spray_data = spray_data[1:]
        while "" in spray_data:
            spray_data.remove("")

        for idx, line in enumerate(spray_data):
            if line == "":
                continue
            key, instances_tensor = extract_instances_tensor(idx=idx, line=line)
            if (
                instances_tensor[spray_label_idx] != 0
                and instances_tensor[spray_label_idx] != 1
            ):
                try:
                    keys_out.remove(key)
                    del data[key]
                except ValueError:
                    pass
            elif key in data:
                data[key][confounder_index] = int(instances_tensor[spray_label_idx])

        if config.spray_groups_balanced:
            _log.info("%s", "re-balancing data group sizes!")
            data_groups = [[], [], [], []]
            data_group_sizes = np.array([0, 0, 0, 0])
            for key in keys_out:
                attribute_tensor = data[key]
                data_group_idx = (
                    attribute_tensor[true_feature_index]
                    + 2 * attribute_tensor[confounder_index]
                ).int()
                data_groups[data_group_idx].append(key)
                data_group_sizes[data_group_idx] += 1

            _log.info("%s %s", "group sizes before re-balancing:", data_group_sizes)
            min_len = np.min(data_group_sizes)
            assert min_len > 0, "need at least one spray label per data group"
            for data_group_idx in {0, 1, 2, 3} - {np.argmin(data_group_sizes).item()}:
                while len(data_groups[data_group_idx]) > min_len:
                    key = data_groups[data_group_idx].pop()
                    del data[key]

            _log.info(
                "%s",
                f"final group sizes: [{len(data_groups[0])}, {len(data_groups[1])}, {len(data_groups[2])}, {len(data_groups[3])}]",
            )
            keys_out = data_groups[0] + data_groups[1] + data_groups[2] + data_groups[3]

    keys_out.sort()
    random.seed(0)
    random.shuffle(keys_out)
    return attributes, data, keys_out


def process_confounder_data_controlled(
    raw_data,
    config,
    mode,
    extract_instances_tensor,
    set_negative_to_zero=True,
):
    """Subsample rows so that target and confounder co-occur at a fixed rate.

    Rows are consumed in file order and sorted into the four (attribute,
    confounder) groups until every group reaches its quota, which is derived
    from ``config.num_samples`` and either ``config.full_confounder_config``
    (four fractions summing to 1) or ``config.confounder_probability`` (the
    aligned groups (0,0) and (1,1) each get ``p/2``, the misaligned ones
    ``(1-p)/2``). Rows with attribute or confounder values >= 2 are skipped.
    Each group is then split with ``config.split`` so that all partitions keep
    the same confounder ratio.

    Parameters
    ----------
    raw_data : iterable
        Rows (csv lines or json records) passed to ``extract_instances_tensor``.
    config : DataConfig
        Provides ``num_samples``, ``split`` and the confounder settings.
    mode : str
        ``"train"``, ``"val"``, ``"test"``; otherwise all selected keys.
    extract_instances_tensor : callable
        ``f(idx, line) -> (key, tensor, attribute, confounder)``.
    set_negative_to_zero : bool, optional
        Clamp negative tensor values (and the attribute/confounder ints) to 0.

    Returns
    -------
    data : dict
        Mapping key -> tensor for every parsed row (not only selected ones).
    keys_out : list
        Selected keys of the requested split, shuffled in place with the
        module-level ``random`` state.

    Raises
    ------
    AssertionError
        If the quotas cannot be filled from the available rows or the totals
        do not add up to ``config.num_samples``.
    """
    data = {}
    n_attribute_confounding = np.array([[0, 0], [0, 0]])
    max_attribute_confounding = np.array([[0, 0], [0, 0]])

    if not config.full_confounder_config is None:
        assert (
            len(config.full_confounder_config) == 4
        ), "confounder config must have 4 entries"
        assert (
            sum(config.full_confounder_config) == 1
        ), "confounder config must sum to 100%"

        max_attribute_confounding[0][0] = int(
            config.num_samples * config.full_confounder_config[0]
        )
        max_attribute_confounding[0][1] = int(
            config.num_samples * config.full_confounder_config[1]
        )
        max_attribute_confounding[1][0] = int(
            config.num_samples * config.full_confounder_config[2]
        )
        max_attribute_confounding[1][1] = int(
            config.num_samples * config.full_confounder_config[3]
        )

    else:
        max_attribute_confounding[0][0] = int(
            config.num_samples * config.confounder_probability * 0.5
        )
        max_attribute_confounding[1][0] = int(
            config.num_samples * round(1 - config.confounder_probability, 3) * 0.5
        )
        max_attribute_confounding[0][1] = int(
            config.num_samples * round(1 - config.confounder_probability, 3) * 0.5
        )
        max_attribute_confounding[1][1] = int(
            config.num_samples * config.confounder_probability * 0.5
        )

    keys = [[[], []], [[], []]]

    for idx, line in enumerate(raw_data):
        if line == "":
            continue

        key, instances_tensor, attribute, confounder = extract_instances_tensor(
            idx=idx, line=line
        )
        if confounder >= 2 or attribute >= 2:
            continue

        if set_negative_to_zero:
            data[key] = torch.maximum(
                torch.zeros_like(instances_tensor),
                instances_tensor,
            )
            confounder = max(0, confounder)
            attribute = max(0, attribute)

        else:
            data[key] = instances_tensor

        if (
            n_attribute_confounding[attribute][confounder]
            < max_attribute_confounding[attribute][confounder]
        ):
            keys[attribute][confounder].append(key)
            n_attribute_confounding[attribute][confounder] += 1

        if np.sum(n_attribute_confounding == max_attribute_confounding) == 4:
            break
    assert (
        np.sum(n_attribute_confounding == max_attribute_confounding) == 4
    ), "something went wrong with filling up the attributes: " + str(
        n_attribute_confounding
    )
    assert (
        np.sum(n_attribute_confounding) == config.num_samples
    ), "wrong number of samples!"
    assert (
        len(keys[0][0]) + len(keys[0][1]) + len(keys[1][0]) + len(keys[1][1])
        == config.num_samples
    ), "wrong number of keys!"
    if mode == "train":
        keys_out = keys[0][0][: int(len(keys[0][0]) * config.split[0])]
        keys_out += keys[0][1][: int(len(keys[0][1]) * config.split[0])]
        keys_out += keys[1][0][: int(len(keys[1][0]) * config.split[0])]
        keys_out += keys[1][1][: int(len(keys[1][1]) * config.split[0])]
        random.shuffle(keys_out)

    elif mode == "val":
        keys_out = keys[0][0][
            int(len(keys[0][0]) * config.split[0]) : int(
                len(keys[0][0]) * config.split[1]
            )
        ]
        keys_out += keys[0][1][
            int(len(keys[0][1]) * config.split[0]) : int(
                len(keys[0][1]) * config.split[1]
            )
        ]
        keys_out += keys[1][0][
            int(len(keys[1][0]) * config.split[0]) : int(
                len(keys[1][0]) * config.split[1]
            )
        ]
        keys_out += keys[1][1][
            int(len(keys[1][1]) * config.split[0]) : int(
                len(keys[1][1]) * config.split[1]
            )
        ]
        random.shuffle(keys_out)

    elif mode == "test":
        keys_out = keys[0][0][int(len(keys[0][0]) * config.split[1]) :]
        keys_out += keys[0][1][int(len(keys[0][1]) * config.split[1]) :]
        keys_out += keys[1][0][int(len(keys[1][0]) * config.split[1]) :]
        keys_out += keys[1][1][int(len(keys[1][1]) * config.split[1]) :]
        random.shuffle(keys_out)

    else:
        keys_out = keys[0][0] + keys[0][1] + keys[1][0] + keys[1][1]
        random.shuffle(keys_out)

    return data, keys_out
