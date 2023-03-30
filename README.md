Welcome to the Pytorch Explain and Adapt Library (PEAL)!

The contribution of this library is two-fold:

1) Explain Neural Network models based on different explanation techniques

2) Adapt Neural Network models according to a given feedback

**Installation Instructions:**

pip install peal

otherwise the project can also be downloaded, a conda or virtualenv environment can be installed based on the requirements.txt (we tested the program for Python 3.9.15) and peal can be used by adding the path to the project to the PYTHONPATH as described in the jupyter notebooks.

**Example Workflow:**

Assuming you have a ```classifier```, a ```dataloader_train```, a ```dataloader_val``` and number of classes ```N```.

```
from peal.adaptors import CounterfactualKnowledgeDistillation

cal = CounterfactualKnowledgeDistillation(
  student = classifier,
  datasource = (dataloader_train, dataloader_val),
  output_size = N,
  teacher = 'human@8000'
)

cal.run()
```

Then the following happens:

1) A folder peal_runs/run1 is created and the classifier is saved under peal_runs/run1/original_model.cpl

2) A generative model is trained based on $PEAL/configs/models/default_generator.yaml and saved under peal_runs/run1/generator .

3) A i'th round of counterfactuals is calculated and the explanation collages under peal_runs/run1/i/collages

4) A web interface is started under localhost:8000 that receives feedback from the user and saves it

5) The counterfactuals are saved with their feedback-adapted label under peal_runs/run1/i/dataset

6) The classifier is finetuned based on $PEAL/configs/training/finetune_classifier.yaml and saved under peal_runs/run1/i/finetuned_model/model.cpl

7) If i smaller then the maximum number of finetune iterations go back to 3.

Generally all configs used can be written yourself and the path can be given via the constructor arguments to the components.

Furthermore, the whole library follows the principle of compositionality, so that arbitrary components in the pipeline can be replaced and the pipeline still works.

$PEAL marks the directory peal of the library where the code and the configs are saved.

More detailed examples that reproduce the results from the paper can be found in the jupyter notebooks.

The Follicle dataset from the paper and how to work with it is propriertary and can thereby not be disclosed in detail here.

**Structure of the Project:**

peal/configs - generic config files, that can be either directly used, extended, adapted or exchanged

peal/templates - original html templates for the feedback webapp

peal/explainers - the different explainers (e.g. counterfactual explanations, layer-wise relevance explanations...)

peal/teachers - the different teachers (e.g. human teacher, oracle teacher, virelay teacher...)

peal/adaptors - the different model adaptors, that are able to refine a model (e.g. counterfactual knowledge distillation, projective class artifact compensation, ...)

peal/architectures - architecture components used for the experiments from the papers and for the generative models necessary to realize CFKD

peal/training - everything that is needed for finetuning models, training generators and train models from experiments

peal/data - datasets, dataloaders and data generators, that e.g. allow creating controlled confounder dataset based on copyright tag

notebooks - the notebooks that walk you step by step how to use the library e.g. to reproduce results of the papers

tests - unit tests to ensure the components work properly on mock data

documentation - the documentation of the project

**Files that are created by executing the jupyter notebooks while reproducing the experiments and would be created in root folder of execution if executed somewhere else**

notebooks/datasets - the folder where the datasets like the augmented CelebA dataset from the experiments are saved

notebooks/templates - the folder for temporary files from the Flask app used for collecting user feedback

notebooks/peal_runs - the folder where the executed runs are stored - some runs can take quite long, so there are checkpoint files as often as possible, that one can restart the the run where interrupted if needed.

**Contribution Guidelines:**

If you want to contribute to the project, please follow the following guidelines:

1) Use the black formatter for python files

2) Avoid code redundancy as much as possible

3) Write unit tests for every new component

4) Write documentation for every new component

5) Write a jupyter notebook that shows how to use the new component

6) Compositionality is key! Try to make the components as independent as possible and make sure that the whole pipeline still works if you replace a component with another one.

7) Always use seeds for every experiment to make sure the results are reproducible

8) Log all important information in the log files to make experiments comparable
