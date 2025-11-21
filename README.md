Welcome to the Pytorch Explain and Adapt Library (PEAL)!

The contribution of this library is two-fold:

1) Explain Neural Network models based on different explanation techniques.

2) Adapt Neural Network models according to a given feedback of a human expert.

If you find this useful, consider citing:

```
@article{bender2025towards,
  title={Towards Desiderata-Driven Design of Visual Counterfactual Explainers},
  author={Bender, Sidney and Herrmann, Jan and M{\"u}ller, Klaus-Robert and Montavon, Gr{\'e}goire},
  journal={arXiv preprint arXiv:2506.14698},
  year={2025}
}
```

**Setup usage with the command line**

First, when using the existing config system, one should be aware of the following system variables that can also be set yourself:

$PEAL_DATA - the path to the data folder, where the datasets are stored, is set by default to "./datasets".
If you download datasets like CelebA or the Follicles dataset, you have to place them in this folder.

$PEAL_RUNS - the path to the runs folder, where the runs are stored, is set by default to "./peal_runs".
If you search for logs, visualizations, or want to use the run for another run, you should search here.

<PEAL_BASE>, the path where the code lies looks a bit like an environment variable as well, but it can be automatically inferred by the library.
Hence, there is no need and also no option to set it yourself.

Then, one should create a conda environment based on the environment.yml file, like:
```conda env create -f environment.yaml``` and activate it with ```conda activate peal```.

An alternative to conda is to work with apptainer by running ```apptainer build python_container.sif python_container.def``` and then run everything inside ```apptainer run --nv python_container.sif```.

**How to no-code use a custom binary image classification dataset with a predictor and a generator from PEAL**

The biggest effort is to reformat the dataset to a ```peal.data.datasets.Image2MixedDataset```.
All labels have to be written into a "$PEAL_DATA/my_data/data.csv" file with the header "imgs,Label1,Label2,...LabelN".
It could also only have one label with "imgs,Label1" and we can only optimize for like this anyway.
All Images have to be placed in the folder "$PEAL_DATA/my_data/imgs" in the correct relative path.
Then, one can copy and adapt the config files for CelebA Smiling as follows:

1) copy configs/sce_experiments/data/celeba.yaml to configs/my_experiments/data/my_data.yaml.
2) copy configs/sce_experiments/data/celeba_generator.yaml to configs/my_experiments/data/my_data_generator.yaml.
3) In both, remove the dataset_class and confounding_factors (because you don't have either for your new dataset yet).
4) In both set dataset_path to "$PEAL_DATA/my_data". You can also set num_samples and output_size, but for this tutorial, it does not matter. Do not change the input_size except if you know what you are doing, because the generative model is restricted in this regard!
5) copy configs/sce_experiments/generators/celeba_ddpm.yaml to configs/my_experiments/generators/my_data_ddpm.yaml.
6) In this file replace base_path with "$PEAL_RUNS/my_data/ddpm" and data with "<PEAL_BASE>/configs/my_experiments/data/my_data_generator.yaml".
7) Train your DDPM generator with: ```python train_generator.py --config "<PEAL_BASE>/configs/my_experiments/generators/my_data_ddpm.yaml"```
8) In parallel, you can copy "configs/sce_experiments/predictors/celeba_Smiling_classifier.yaml" to "configs/my_experiments/predictors/my_data_classifier.yaml".
9) Here, you have to replace model_path with "$PEAL_RUNS/my_data/classifier", data with p and y_selection with "[Label1]".
10) Now you can train your predictor with: ```python train_predictor.py --config "<PEAL_BASE>/configs/my_experiments/predictors/my_data_classifier.yaml"```
11) After finishing generator and predictor training, you can copy "configs/sce_experiments/adaptors/celeba_Smiling_natural_sce_cfkd.yaml" 
to "configs/my_experiments/adaptors/my_data_sce_cfkd.yaml".
12) Overwrite data with "<PEAL_BASE>/configs/my_experiments/data/my_data.yaml", student with "$PEAL_RUNS/my_data/classifier/model.cpl", generator with "$PEAL_RUNS/my_data/ddpm/config.yaml", base_dir with "$PEAL_RUNS/my_data/classifier/sce_cfkd" and y_selection with "[Label1]" and calculate_explainer_stats with "False".
13) Now you can run SCE with: ```python run_cfkd.py --config "<PEAL_BASE>/configs/my_experiments/adaptors/my_data_sce_cfkd.yaml"```
14) Now you can find your most salient counterfactuals under "$PEAL_RUNS/my_data/classifier/sce_cfkd/validation_collages0_1".
15) You can find secondary counterfactuals under "$PEAL_RUNS/my_data/classifier/sce_cfkd/validation_collages0_0", but it might be possible that they look destroyed if SCE could not find another counterfactual and forced it too much

If you further want to process them, you can load the .npz array "$PEAL_RUNS/my_data/classifier/sce_cfkd/validation_tracked_values.npz".
The originals in this array can be found under the key "x_list" and the counterfactuals under "x_counterfactual_list".


**How to use a custom image dataset and a custom predictor**


First, you can copy "configs/sce_experiments/adaptors/imagenet_husky_vs_wulf_sce_cfkd.yaml" to "configs/my_experiments/adaptors/my_data_sce_cfkd.yaml".

If you do not wish to use the ImageNet DDPM as generative model one has to copy "configs/sce_experiments/data/imagenet_generator.yaml" to "configs/data/my_data_generator.yaml".
Then, the dataset_class in "configs/my_experiments/data/my_data_generator.yaml" has to be removed.
Then, num_samples, input_size and output_size should be adapted to your dataset.
If you do not want to bootstrap your DDPM with Imagenet 256x256 weights you have to remove download_weights.
Now, one has to copy configs/sce_experiments/generators/imagenet_ddpm.yaml to configs/my_experiments/generators/my_data_ddpm.yaml.
In this file one has to replace base_path with "$PEAL_RUNS/my_data/ddpm" and data with "<PEAL_BASE>/configs/my_experiments/data/my_data_generator.yaml".
Now you can train your DDPM generator with:
```python train_generator.py --config "<PEAL_BASE>/configs/my_experiments/generators/my_data_ddpm.yaml"```
Then, in "configs/my_experiments/adaptors/my_data_sce_cfkd.yaml" one has to overwrite generator with "$PEAL_RUNS/my_data_ddpm.yaml".

Next, you have to convert your predictor into an ONNX model with binary output.
An example for an ImageNet classifier would be the following:

```
import torch
import os
import torchvision

class BinaryImageNetModel(torch.nn.Module):
    def __init__(self, class1, class2):
        super(BinaryImageNetModel, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.class1 = class1
        self.class2 = class2

    def forward(self, x):
        logits_full = self.model(x)
        return logits_full[:, [self.class1, self.class2]]

wulf_vs_husky_classifier = BinaryImageNetModel(248, 269)
wulf_vs_husky_classifier.eval()

os.makedirs("$PEAL_RUNS/imagenet")
os.makedirs("$PEAL_RUNS/imagenet/wulf_vs_husky_classifier")
dummy_input = torch.randn(1, 3, 224, 224)  # standard ResNet input
OUTPUT_PATH = "$PEAL_RUNS/wulf_vs_husky_classifier/model.onnx"
torch.onnx.export(
    wulf_vs_husky_classifier,
    dummy_input,
    OUTPUT_PATH,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)
```
You have to create an equivalent code snippet for your model and save it under OUTPUT_PATH="$PEAL_RUNS/my_data/classifier/model.onnx".
Now in "configs/my_experiments/adaptors/my_data_sce_cfkd.yaml" one has to overwrite student with "$PEAL_RUNS/my_data/classifier/model.onnx"
and base_dir with "$PEAL_RUNS/my_data/classifier/sce_cfkd".

Now you have to reformat your dataset to a ```peal.data.datasets.Image2MixedDataset```.
All labels have to be written into a "<PEAL_DATA>/my_data/data.csv" file with the header "ImagePath,Label1,Label2,...LabelN".
It could also only have one label with "ImagePath,Label1" and we can only optimize for like this anyway.
In the case of the ImageNet wulf vs husky task, one can use the header "ImagePath,IsWulf" and now adds all relative paths to
images of huskies and wulfs to get a csv in the format:

| ImagePath           | IsWulf |
|---------------------|--------|
| path_to_husky_0.png | 0      |
| path_to_husky_1.png | 0      |
| ...                 | ...    |
| path_to_husky_N.png | 0      |
| path_to_wulf_0.png  | 1      |
| path_to_wulf_1.png  | 1      |
| ...                 | ...    |
| path_to_wulf_N.png  | 1      |

Next, one has to copy "configs/sce_experiments/data/imagenet.yaml" to "configs/my_experiments/data/my_data.yaml".
In the new file you have to change input_size and normalization to the values your classifier was trained with.
Then, in "configs/my_experiments/adaptors/my_data_sce_cfkd.yaml" one has to overwrite data with "<PEAL_BASE>/configs/my_experiments/data/my_data.yaml", and
y_selection with "[Label1]".

Next, you can run SCE with:
```python run_cfkd.py --config "<PEAL_BASE>/configs/my_experiments/adaptors/my_data_sce_cfkd.yaml"```
Now you can find your counterfactuals under "$PEAL_RUNS/my_data/classifier/sce_cfkd/validation_collages".
If you further want to process them, you can load the .npz array "$PEAL_RUNS/my_data/classifier/sce_cfkd/validation_tracked_values.npz".
The originals in this array can be found under the key "x_list" and the counterfactuals under "x_counterfactual_list".

**Execute your own code based on PEAL components:**

You can utilize different top-level scripts to use the library:

```python generate_dataset.py --config "<PATH_TO_CONFIG>"```
Generates a synthetic dataset like the squares dataset based on a data config file.
Another option is to augment a already downloaded dataset like CelebA with a controlled confounder.
Existing data config files can be found in configs/sce_experiments/data.
The generated dataset will appear in $PEAL_DATA.
For the replication of the experiments on synthetic / augmented datasets from the paper one first needs to generate the articial datasets.
A full, documented overview over the parameters that can be set by datasets so far can be found in ```peal.data.datasets.DataConfig```.
To implement a new dataset ```MyDataset``` create a subclass of ```peal.data.interfaces.PealDataset``` inside ```peal.data```.
Then set ```dataset_class="MyDataset"``` in your data config file and ```MyDataset``` will be used automatically.
However, a lot of ImageDataset like MNIST, CIFAR-10, ImageNet etc can be already expressed via the ```peal.data.datasets.Image2ClassDataset```.
Moreover, a lot of datasets like CelebA, Squares, Follicles or Regression dataset can be already expressed via the ```peal.data.datasets.Image2MixedDataset```.

```python train_predictor.py --config "<PATH_TO_CONFIG>"```
Trains a predictor (either a singleclass classifier, a multiclass classifier, a regressor or a mixed model) based on a predictor config file.
Existing predictor config files can be found in configs/sce_experiments/predictors.
In order to replicate most experiments from the papers one needs to train a student predictor and a teacher predictor.
Trained predictors will appear in $PEAL_RUNS.
A full, documented overview over the parameters that can be set by predictors so far can be found in ```peal.training.trainers.PredictorConfig```.

```python train_generator.py --config "<PATH_TO_CONFIG>"```
Trains a generative model based on a generator config file.
Existing generator config files can be found in configs/sce_experiments/models.
In order to e.g. use counterfactual explainations one needs to train a corresponding generator.
Trained generators will appear in $PEAL_RUNS.
A full, documented overview over the parameters that can be set by generators so far can be found in subclasses of ```peal.generators.interfaces.GeneratorConfig``` in ```peal.generators```.
To implement a new generator ```MyGenerator``` create a subclass of either ```peal.generators.interfaces.InvertibleGenerator``` or of ```peal.generators.interfaces.EditCapableGenerator``` inside ```peal.generators```.
Then create a new subclass ```MyGeneratorConfig``` of ```peal.generators.interfaces.GeneratorConfig``` inside ```peal.generators```.
Set ```generator_type="MyGenerator"``` in ```MyGeneratorConfig```.
Now ```MyGenerator``` will be used automatically and configured by ```MyGeneratorConfig``` while initialization.

```python run_explainer.py --config "<PATH_TO_CONFIG>"```
Explains the predictions of a predictor based on a explainer config file.
Existing explainer config files can be found in configs/sce_experiments/explainers.
The results of the explanations will be visualized in a web interface.
Furthermore, they are saved in a folder inside the folder where the predictor that was explained is saved.
A full, documented overview over the parameters that can be set by explainers so far can be found in subclasses of ```peal.explainers.interfaces.ExplainerConfig``` in ```peal.explainers```.
To implement a new explainer ```MyExplainer``` create a subclass of ```peal.explainers.interfaces.Explainer``` inside ```peal.explainers```.
Then create a new subclass ```MyExplainerConfig``` of ```peal.explainers.interfaces.ExplainerConfig``` inside ```peal.explainers```.
Set ```explainer_type="MyExplainer"``` in ```MyExplainerConfig```.
Now ```MyExplainer``` will be used automatically and configured by ```MyExplainerConfig``` while initialization.


```python run_adaptor.py --config "<PATH_TO_CONFIG>"```
Adapts a predictor based on a adaptor config file.
Existing adaptor config files can be found in configs/sce_experiments/adaptors.
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

**Example Workflow CFKD adaptor:**

Here we introduce a code snippet that does the same as "run_cfkd.py" does in a no-code CLI manner.
Assuming you have a predictor ```my_classifier``` and a dataset ```my_dataset``` and a peal.adaptor.counterfactual_knowledge_distillation.CFKDConfig ```adaptor_config``` configuring your CFKD run.

```
from peal.adaptors.counterfactual_knowledge_distillion import CFKD

cfkd = CFKD(
  student = my_classifier,
  datasource = my_dataset,
  teacher = 'human@8000',
  adaptor_config = adaptor_config,
)

fixed_classifier = cfkd.run()
```

Then the following happens:

1) A folder peal_runs/run1 is created and the classifier is saved under peal_runs/run1/original_model.cpl

2) A generative model is trained based on $PEAL/configs/models/default_generator.yaml and saved under peal_runs/run1/generator .

3) A i'th round of counterfactuals is calculated and the explanation collages under peal_runs/run1/i/collages

4) A web interface is started under localhost:8000 that receives feedback from the user and saves it

5) The counterfactuals are saved with their feedback-adapted label under peal_runs/run1/i/dataset

6) The classifier is finetuned based on $PEAL/configs/training/finetune_classifier.yaml and saved under peal_runs/run1/i/finetuned_model/model.cpl

7) If i smaller then the maximum number of finetune iterations go back to 3.


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

1) Use the black formatter with 88 columns for python files

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
