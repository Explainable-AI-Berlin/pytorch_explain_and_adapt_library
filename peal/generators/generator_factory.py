from peal.generators.interfaces import InvertibleGenerator
from peal.generators.normalizing_flows import Glow
from peal.utils import load_yaml_config
from peal.training.trainers import ModelTrainer


def get_generator(generator, data_config, train_dataloader, dataloaders_val, base_dir, gigabyte_vram, device):
    '''
    _summary_

    Args:
        generator (_type_): _description_
        adaptor_config (_type_): _description_
        train_dataloader (_type_): _description_
        dataloaders_val (_type_): _description_
        base_dir (_type_): _description_
        gigabyte_vram (_type_): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_
    '''
    if isinstance(generator, InvertibleGenerator):
        generator = generator

    else:
        generator_config = load_yaml_config(generator)
        generator_config["data"] = data_config
        generator = Glow(generator_config).to(device)
        generator_trainer = ModelTrainer(
            config=generator_config,
            model=generator,
            datasource=(
                train_dataloader.dataset,
                dataloaders_val[0].dataset,
            ),
            base_dir=base_dir,
            model_name="generator",
            gigabyte_vram=gigabyte_vram,
        )
        print("Train generator model!")
        generator_trainer.fit()

    generator.eval()

    return generator
