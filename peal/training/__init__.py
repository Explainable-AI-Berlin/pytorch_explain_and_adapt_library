"""Training of predictors: trainers, criterions, loggers and their configs.

``interfaces`` defines ``TrainingConfig`` and ``PredictorConfig`` (the yaml
schema of a predictor run); ``trainers.ModelTrainer`` runs the epoch loop
with optional adversarial, mixup and diffusion augmentation and also hosts
the distillation helpers CFKD and DiDAE rely on; ``criterions`` maps the
``task.criterions`` keys to loss functions; ``loggers`` writes per-step and
per-epoch metrics to TensorBoard and ``training_utils`` collects small
shared helpers.
"""
