Welcome to the Pytorch Explain and Adapt Library (PEAL)!

The contribution of this library is two-fold:

1) Explain Neural Network models based on different explanation techniques.

2) Adapt Neural Network models according to a given feedback of a human expert.

**Usage with the command line**

First in order when using the existing config system, one should be aware of the following system variabales used that can be also set yourself:

$PEAL_DATA - the path to the data folder, where the datasets are stored, is set per default to "./datasets".
If you download datasets like CelebA or the Follicles dataset you have to place it in this folder.

$PEAL_RUNS - the path to the runs folder, where the runs are stored, is set per default to "./peal_runs".
If you search for logs, visualizations or want to use the run for another run you should search here.

<PEAL_BASE>, the path where the code lies looks a bit like a environment variable as well, but it can be automatically inferred by the library.
Hence, there is no need and also no option to set it yourself.

Then, one should create a conda environment based on the environment.yml file like:
```conda env create -f environment.yml``` and activate it with ```conda activate peal```.

An alternative to conda is to work with apptainer by running ```apptainer build python_container.sif python_container.def``` and then run everything inside ```apptainer run --nv python_container.sif```.

Now one can utilize different top level scripts to use the library:

```python generate_dataset.py --config "<PATH_TO_CONFIG>"```
Generates a synthetic dataset like the squares dataset based on a data config file.
Another option is to augment a already downloaded dataset like CelebA with a controlled confounder.
Existing data config files can be found in configs/data.
The generated dataset will appear in $PEAL_DATA.
For the replication of the experiments on synthetic / augmented datasets from the paper one first needs to generate the articial datasets.
A full, documented overview over the parameters that can be set by datasets so far can be found in ```peal.data.datasets.DataConfig```.
To implement a new dataset ```MyDataset``` create a subclass of ```peal.data.interfaces.PealDataset``` inside ```peal.data```.
Then set ```dataset_class="MyDataset"``` in your data config file and ```MyDataset``` will be used automatically.
However, a lot of ImageDataset like MNIST, CIFAR-10, ImageNet etc can be already expressed via the ```peal.data.datasets.Image2ClassDataset```.
Moreover, a lot of datasets like CelebA, Squares, Follicles or Regression dataset can be already expressed via the ```peal.data.datasets.Image2MixedDataset```.

```python train_predictor.py --config "<PATH_TO_CONFIG>"```
Trains a predictor (either a singleclass classifier, a multiclass classifier, a regressor or a mixed model) based on a predictor config file.
Existing predictor config files can be found in configs/predictors.
In order to replicate most experiments from the papers one needs to train a student predictor and a teacher predictor.
Trained predictors will appear in $PEAL_RUNS.
A full, documented overview over the parameters that can be set by predictors so far can be found in ```peal.architectures.PredictorConfig```.

```python train_generator.py --config "<PATH_TO_CONFIG>"```
Trains a generative model based on a generator config file.
Existing generator config files can be found in configs/models.
In order to e.g. use counterfactual explainations one needs to train a corresponding generator.
Trained generators will appear in $PEAL_RUNS.
A full, documented overview over the parameters that can be set by generators so far can be found in subclasses of ```peal.generators.interfaces.GeneratorConfig``` in ```peal.generators```.
To implement a new generator ```MyGenerator``` create a subclass of either ```peal.generators.interfaces.InvertibleGenerator``` or of ```peal.generators.interfaces.EditCapableGenerator``` inside ```peal.generators```.
Then create a new subclass ```MyGeneratorConfig``` of ```peal.generators.interfaces.GeneratorConfig``` inside ```peal.generators```.
Set ```generator_type="MyGenerator"``` in ```MyGeneratorConfig```.
Now ```MyGenerator``` will be used automatically and configured by ```MyGeneratorConfig``` while initialization.

```python run_explainer.py --config "<PATH_TO_CONFIG>"```
Explains the predictions of a predictor based on a explainer config file.
Existing explainer config files can be found in configs/explainers.
The results of the explanations will be visualized in a web interface.
Furthermore, they are saved in a folder inside the folder where the predictor that was explained is saved.
A full, documented overview over the parameters that can be set by explainers so far can be found in subclasses of ```peal.explainers.interfaces.ExplainerConfig``` in ```peal.explainers```.
To implement a new explainer ```MyExplainer``` create a subclass of ```peal.explainers.interfaces.Explainer``` inside ```peal.explainers```.
Then create a new subclass ```MyExplainerConfig``` of ```peal.explainers.interfaces.ExplainerConfig``` inside ```peal.explainers```.
Set ```explainer_type="MyExplainer"``` in ```MyExplainerConfig```.
Now ```MyExplainer``` will be used automatically and configured by ```MyExplainerConfig``` while initialization.


```python run_adaptor.py --config "<PATH_TO_CONFIG>"```
Adapts a predictor based on a adaptor config file.
Existing adaptor config files can be found in configs/adaptors.
The results of the explanations will be saved in a folder inside the folder where the predictor that was adapted is saved.
A full, documented overview over the parameters that can be set by adaptors so far can be found in subclasses of ```peal.adaptors.interfaces.AdaptorConfig``` in ```peal.adaptors```.
To implement a new adaptor ```MyAdaptor``` create a subclass of ```peal.adaptors.interfaces.Adaptor``` inside ```peal.adaptors```.
Then create a new subclass ```MyAdaptorConfig``` of ```peal.adaptors.interfaces.AdaptorConfig``` inside ```peal.adaptors```.
Set ```adaptor_type="MyAdaptor"``` in ```MyAdaptorConfig```.
Now ```MyAdaptor``` will be used automatically and configured by ```MyAdaptorConfig``` while initialization.


Hint: All configuration is done via Pydantic.
Hence, the config files can be given as YAML files, but will be parsed as Python objects.
In this process only the values that are set in the YAML file are overwritten in the Python template, the rest of the values will stay at the default.
The documentation can be found in the corresponding Python classes in the code.

<!---
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
-->

**Structure of the Project:**

peal/explainers - the different explainers (e.g. counterfactual explanations, layer-wise relevance explanations...)

peal/teachers - the different teachers (e.g. human teacher, oracle teacher, segmentation mask teacher...)

peal/adaptors - the different model adaptors, that are able to refine a model (e.g. counterfactual knowledge distillation, projective class artifact compensation, ...)

peal/architectures - architecture components used for the available predictors

peal/training - everything that is needed for training and finetuning predictors

peal/data - datasets, dataloaders and data generators, that e.g. allow creating controlled confounder dataset based on copyright tag

peal/generators - the different generative models that can be used to generate counterfactuals

peal/dependencies - integration of related work that does not provide a library

configs - generic config files, that can be either directly used, extended, adapted or exchanged

notebooks - the notebooks that walk you step by step how to use the library e.g. to reproduce results of the papers

templates - original html templates for the feedback webapp

tests - unit tests to ensure the components work properly on mock data

docs - The html documentation of the project generated by Sphinx.
Has to be watched in web browser directly to make sense

**Contribution Guidelines:**

If you want to contribute to the project, please follow the following guidelines:

1) Use the black formatter for python files

2) Avoid code redundancy as much as possible

3) Write unit tests for every new component that have high code coverage

4) Write sphinx parsable documentation for every new component in the NumPy documentation style

5) Extend existing code instead of changing it!!!

6) Compositionality is key! Try to make the components as independent as possible and make sure that the whole pipeline still works if you replace a component with another one.

7) Always use seeds for every experiment to make sure the results are exactly reproducible

8) Log all important information in the log files to make experiments comparable

9) No classes longer than 500 lines of code and no methods longer than 100 lines of code

10) Use design patterns like the factory or visitor pattern to make the code more readable and maintainable

11) Benchmark your code that gpu utilization is high enough instead of being to busy with loading data or moving it between cpu and gpu
