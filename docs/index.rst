PEAL: PyTorch Explain and Adapt Library
=======================================

PEAL explains image classifiers with visual counterfactuals and repairs the
Clever-Hans strategies those counterfactuals expose. It is the official
implementation of the Smoothed Counterfactual Explorer (SCE), Counterfactual
Knowledge Distillation (CFKD) and Disentangled Diffusion Autoencoders (DiDAE),
and ships re-implementations of the counterfactual explainers and
robustification baselines those papers compare against.

PEAL can be used three ways, all driven by the same config files:

* **Command line** -- ``peal-cfkd``, ``peal-didae``, ``peal-train-predictor`` and the
  other commands each run one method from a config file. This is also how the papers'
  results are reproduced (``reproduction_scripts/``).
* **Python API** -- ``import peal`` gives the config models, factories and methods
  (``peal.CFKD``, ``peal.DiDAE``, ...); see the :doc:`API reference <reference/index>`.
* **Web interface** -- upload an ONNX classifier and a dataset, judge the
  counterfactuals in the browser and download the corrected model.

The :doc:`user guide <usage>` walks through all three.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   architecture
   usage

.. toctree::
   :maxdepth: 1
   :caption: Model cards

   model_cards/rae_clip_imagenet

.. toctree::
   :maxdepth: 1
   :caption: Development

   development

.. toctree::
   :maxdepth: 2
   :caption: API reference

   reference/index

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
