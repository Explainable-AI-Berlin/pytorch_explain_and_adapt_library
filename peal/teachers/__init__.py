"""Teachers: the sources of feedback an adaptor learns from.

A teacher implements ``interfaces.TeacherInterface.get_feedback`` and labels
the explanations an explainer produced (typically: did the counterfactual
change the true feature, a spurious one, or leave the data manifold?).
Included are a human in a Flask GUI (``human2model_teacher``), a headless
Claude process (``llm2model_teacher``), another model with optional label
noise (``model2model_teacher``), segmentation masks, symbolic rules, the
SpRAy / ViRelAy clustering tools, direction-level labelling for DiDAE
(``cluster_teacher``, ``preclustered_teacher``), the web-demo feedback bridge
and a no-feedback baseline. ``teacher_factory.get_teacher`` resolves the
``teacher`` config string to one of them.
"""
