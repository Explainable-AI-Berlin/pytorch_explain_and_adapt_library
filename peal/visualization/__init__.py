"""Rendering helpers the pipeline uses, plus the DiDAE table-1 script.

``image_grid`` and ``model_comparison`` are imported by the pipeline itself
(CFKD's collages and the LRP comparison), ``visualize_counterfactual_gradients``
by the counterfactual explainer, and ``create_didae_table1`` reads the result
yamls a DiDAE run leaves under ``$PEAL_RUNS`` into the paper's table 1.
"""
