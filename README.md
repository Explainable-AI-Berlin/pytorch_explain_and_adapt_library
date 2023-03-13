Welcome to the Pytorch Explain and Adapt Library (PEAL)!

The contribution of this library is two-fold:

1) Explain Neural Network models based on different explanation techniques

2) Adapt Neural Network models according to given feedback

**Installation Instructions:**

pip install peal

**Example Workflow:**

Assuming you have a ```classifier```, a ```dataloader_train``` and a ```dataloader_val``` with input channels ```C```,  input height ```H```,  input width ```W``` and number of classes ```N```.

```
from peal.adaptors import CounterfactualKnowledgeDistillation

cal = CounterfactualKnowledgeDistillation(
  student = classifier,
  datasource = (dataloader_train, dataloader_val),
  input_size = [C,H,W],
  output_size = N,
  run_name = 'teaching_my_classifer',
  teacher = 'Human@8000'
)

cal.run()
```

Then the following happens:

1) A folder peal_runs/run1 is created and the classifier is saved under peal_runs/teaching_my_classifer/original_model.cpl

2) A generative model is trained based on $PEAL/configs/models/default_generator.yaml and saved under peal_runs/teaching_my_classifer/generator .

3) A i'th round of counterfactuals is calculated and the explanation collages under peal_runs/teaching_my_classifer/i/collages

4) A web interface is started under localhost:8000 that receives feedback from the user and saves it

5) The counterfactuals are saved with their potentially based on the feedback adapted label under peal_runs/teaching_my_classifer/i/dataset

6) The classifier is finetuned based on $PEAL/configs/training/finetune_classifier.yaml and saved under peal_runs/teaching_my_classifer/i/finetuned_model/model.cpl

7) If i smaller then the maximum number of finetune iterations go back to 3.

Generally all configs used can be written yourself and the path can be given via as command to the adaptor.

Furthermore, the whole library follows the principle of compositionality, so that arbitrary components in the pipeline can be replaced and the pipeline still works.

$PEAL marks the directory peal of the library where the code and the configs are saved.

**Structure of the Project:**

peal/configs - generic config files, that can be either used or adapted

peal/templates - html files for the feedback webapp

peal/explainers - the different explainers (e.g. counterfactual explanations)

peal/adaptors - the different model adaptors, that are able to refine a model (e.g. counterfactual knowledge distillation)

peal/architectures - architectures used for the experiments from the paper and for the generators

peal/training - everything that is needed for finetuning models, training generators and train models from experiments

peal/data - datasets, dataloaders and data generators, that e.g. allow creating controlled confounder dataset based on copyright tag

notebooks - the notebooks that walk you step by step how to use the library e.g. to reproduce results of the papers
