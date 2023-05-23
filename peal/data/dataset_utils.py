import json
import torch
import random
import numpy as np


def parse_json(data_dir, config, mode, set_negative_to_zero=True):
    """
    _summary_

    Args:
        data_dir (_type_): _description_
        config (_type_): _description_
        mode (_type_): _description_

    Returns:
        _type_: _description_
    """
    with open(data_dir, "r") as f:
        raw_data = json.load(f)

    if config["known_confounder"]:

        def extract_instances_tensor_confounder(idx, line):
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


def parse_csv(data_dir, config, mode, key_type="idx", set_negative_to_zero=True):
    """
    _summary_

    Args:
        data_dir (_type_): _description_
        config (_type_): _description_
        mode (_type_): _description_
        key_type (str, optional): _description_. Defaults to "idx".

    Returns:
        _type_: _description_
    """
    raw_data = open(data_dir, "r").read().split("\n")
    attributes = raw_data[0].split(",")
    if key_type == "name":
        attributes = attributes[1:]

    raw_data = raw_data[1:]

    def extract_instances_tensor(idx, line):
        instance_attributes = line.split(",")

        if key_type == "idx":
            key = str(idx)

        elif key_type == "name":
            key = instance_attributes[0]
            instance_attributes = instance_attributes[1:]

        instance_attributes.remove("") if "" in instance_attributes else None
        instance_attributes_int = list(
            map(
                lambda x: float(x),
                instance_attributes,
            )
        )
        instances_tensor = torch.tensor(instance_attributes_int)
        return key, instances_tensor

    if not len(config["confounding_factors"]) == 0:

        def extract_instances_tensor_confounder(idx, line):
            selection_idx1 = attributes.index(config["confounding_factors"][0])
            selection_idx2 = attributes.index(config["confounding_factors"][1])
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
        for idx, line in enumerate(raw_data):
            key, instances_tensor = extract_instances_tensor(idx, line)
            data[key] = torch.maximum(
                torch.zeros_like(instances_tensor),
                instances_tensor,
            )

        keys = list(data.keys())
        if mode == "train":
            keys_out = keys[: int(len(keys) * config["split"][0])]

        elif mode == "val":
            keys_out = keys[
                int(len(keys) * config["split"][0]) : int(
                    len(keys) * config["split"][1]
                )
            ]

        elif mode == "test":
            keys_out = keys[int(len(keys) * config["split"][1]) :]

        else:
            keys_out = keys

    return attributes, data, keys_out


def process_confounder_data_controlled(
    raw_data,
    config,
    mode,
    extract_instances_tensor,
    set_negative_to_zero=True,
):
    """
    _summary_

    Args:
        raw_data (_type_): _description_
        config (_type_): _description_
        mode (_type_): _description_
        extract_instances_tensor (_type_): _description_

    Returns:
        _type_: _description_
    """
    data = {}
    n_attribute_confounding = np.array([[0, 0], [0, 0]])
    max_attribute_confounding = np.array([[0, 0], [0, 0]])
    max_attribute_confounding[0][0] = int(
        config["num_samples"] * config["confounder_probability"] * 0.5
    )
    max_attribute_confounding[1][0] = int(
        config["num_samples"] * round(1 - config["confounder_probability"], 2) * 0.5
    )
    max_attribute_confounding[0][1] = int(
        config["num_samples"] * round(1 - config["confounder_probability"], 2) * 0.5
    )
    max_attribute_confounding[1][1] = int(
        config["num_samples"] * config["confounder_probability"] * 0.5
    )
    keys = [[[], []], [[], []]]

    for idx, line in enumerate(raw_data):
        if line == "":
            continue

        key, instances_tensor, attribute, confounder = extract_instances_tensor(
            idx=idx, line=line
        )
        if (
            n_attribute_confounding[attribute][confounder]
            < max_attribute_confounding[attribute][confounder]
        ):
            if set_negative_to_zero:
                data[key] = torch.maximum(
                    torch.zeros_like(instances_tensor),
                    instances_tensor,
                )

            else:
                data[key] = instances_tensor

            keys[attribute][confounder].append(key)
            n_attribute_confounding[attribute][confounder] += 1

        if np.sum(n_attribute_confounding == max_attribute_confounding) == 4:
            break

    assert (
        np.sum(n_attribute_confounding == max_attribute_confounding) == 4
    ), "something went wrong with filling up the attributes"
    assert (
        np.sum(n_attribute_confounding) == config["num_samples"]
    ), "wrong number of samples!"
    assert (
        len(keys[0][0]) + len(keys[0][1]) + len(keys[1][0]) + len(keys[1][1])
        == config["num_samples"]
    ), "wrong number of keys!"
    if mode == "train":
        keys_out = keys[0][0][: int(len(keys[0][0]) * config["split"][0])]
        keys_out += keys[0][1][: int(len(keys[0][1]) * config["split"][0])]
        keys_out += keys[1][0][: int(len(keys[1][0]) * config["split"][0])]
        keys_out += keys[1][1][: int(len(keys[1][1]) * config["split"][0])]
        random.shuffle(keys_out)

    elif mode == "val":
        keys_out = keys[0][0][
            int(len(keys[0][0]) * config["split"][0]) : int(
                len(keys[0][0]) * config["split"][1]
            )
        ]
        keys_out += keys[0][1][
            int(len(keys[0][1]) * config["split"][0]) : int(
                len(keys[0][1]) * config["split"][1]
            )
        ]
        keys_out += keys[1][0][
            int(len(keys[1][0]) * config["split"][0]) : int(
                len(keys[1][0]) * config["split"][1]
            )
        ]
        keys_out += keys[1][1][
            int(len(keys[1][1]) * config["split"][0]) : int(
                len(keys[1][1]) * config["split"][1]
            )
        ]
        random.shuffle(keys_out)

    elif mode == "test":
        keys_out = keys[0][0][int(len(keys[0][0]) * config["split"][1]) :]
        keys_out += keys[0][1][int(len(keys[0][1]) * config["split"][1]) :]
        keys_out += keys[1][0][int(len(keys[1][0]) * config["split"][1]) :]
        keys_out += keys[1][1][int(len(keys[1][1]) * config["split"][1]) :]
        random.shuffle(keys_out)

    else:
        keys_out = keys[0][0] + keys[0][1] + keys[1][0] + keys[1][1]
        random.shuffle(keys_out)

    return data, keys_out
