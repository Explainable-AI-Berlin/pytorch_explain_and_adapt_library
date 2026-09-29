def validate_stage1_config(config) -> None:
    if not config.stage_1.target:
        raise ValueError("Config must provide a 'stage_1' section with target.")
    if not config.gan.loss:
        raise ValueError("Config must define a top-level 'gan' section.")
    valid_dataset_types = {"hf", "histo_manifest"}
    if config.dataset.type not in valid_dataset_types:
        raise ValueError(f"dataset.type must be one of {sorted(valid_dataset_types)}, got '{config.dataset.type}'")
