"""``ModelTrainer`` trains a toy predictor of each modality for one epoch.

This is the deepest test of the set: it goes through ``get_predictor``,
``create_dataloaders_from_datasource``, the criterions, the ``Logger`` and
``fit`` with non-image data, and checks the run directory layout every other
PEAL component reads (``model.cpl``, ``config.yaml``, ``checkpoints/``).
"""

import os

import pytest
import torch

from tests.modalities import toys


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_model_trainer_fits_and_writes_the_run_directory(domain, datasets, tmp_path):
    from peal.architectures.interfaces import TaskConfig
    from peal.training.interfaces import PredictorConfig, TrainingConfig
    from peal.training.trainers import ModelTrainer

    run_dir = str(tmp_path / "run")
    config = PredictorConfig(
        training=TrainingConfig(
            max_epochs=1,
            train_batch_size=8,
            val_batch_size=4,
            test_batch_size=4,
            learning_rate=0.01,
            optimizer="adam",
        ),
        task=TaskConfig(
            criterions={"ce": 1.0}, output_type="singleclass", output_channels=2
        ),
        data=toys.data_config(domain),
        model_path=run_dir,
        tracking_level=1,
    )
    trainer = ModelTrainer(
        config,
        model=domain.predictor(),
        datasource=datasets,
        unit_test_train_loop=True,
    )
    trainer.fit()
    assert os.path.isfile(os.path.join(run_dir, "model.cpl"))
    assert os.path.isfile(os.path.join(run_dir, "config.yaml"))
    assert os.path.isdir(os.path.join(run_dir, "checkpoints"))
    model = torch.load(os.path.join(run_dir, "model.cpl"), weights_only=False)
    x = torch.stack([datasets[2][i][0] for i in range(4)])
    assert model(x).shape == (4, 2)
