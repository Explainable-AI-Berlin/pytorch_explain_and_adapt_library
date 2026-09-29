"""Adaptors: the procedures that repair a predictor from explanation feedback.

An adaptor takes a trained predictor and a source of feedback (a teacher) and
returns a corrected predictor. This subpackage holds CFKD (counterfactual
knowledge distillation on teacher-labelled counterfactuals), DiDAE (ranking of
sparse-dictionary directions in a diffusion autoencoder's latent space,
followed by CFKD on the directions labelled false), the ClArC / P-ClArC /
RR-ClArC concept projections, several Group-DRO variants and a projection
adaptor. ``adaptor_factory.get_adaptor`` builds one from an ``AdaptorConfig``
yaml; every adaptor subclasses ``interfaces.Adaptor``.
"""
