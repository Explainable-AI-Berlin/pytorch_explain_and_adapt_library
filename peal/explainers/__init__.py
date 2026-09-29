"""Explainers: methods that explain a predictor's decisions.

The counterfactual explainers in ``counterfactual_explainer`` (SCE, ACE,
TIME, DAE-distill and the perfect-false-counterfactual oracle) pair a
predictor with a generator from ``peal.generators`` and search for minimal
edits that flip the prediction; ``no_generator_counterfactual_explainers``
(DiCE) works on tabular data without a generator and ``lrp_explainer`` yields
attribution heatmaps. ``explainer_factory.get_explainer`` instantiates one
from an ``ExplainerConfig``; adaptors such as CFKD call ``explain_batch`` to
collect the explanations a teacher then labels.
"""
