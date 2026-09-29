Welcome to the Pytorch Explain and Adapt Library (PEAL)!

The contribution of this library is three-fold:

1) Official implementation of the **Smoothed Counterfactual Explorer (SCE)**.

2) Official implementation of **Counterfactual Knowledge Distillation (CFKD)**.

3) Official implementation of **Disentangled Diffusion Autoencoders (DiDAE)**.

Additionally, the library includes reimplementations of other commonly used counterfactual explainers and model robustification techniques.

If you find this useful or use SCE (https://arxiv.org/pdf/2506.14698), consider citing:

```
@article{bender2025towards,
  title={Towards Desiderata-Driven Design of Visual Counterfactual Explainers},
  author={Bender, Sidney and Herrmann, Jan and M{\"u}ller, Klaus-Robert and Montavon, Gr{\'e}goire},
  journal={arXiv preprint arXiv:2506.14698},
  year={2025}
}
```
You can reproduce the results from the paper with reproduction_scripts/reproduce_sce_results.sh.

If you use CFKD (https://arxiv.org/pdf/2510.17524) useful consider citing:

```
@article{bender2025mitigating,
  title={Mitigating Clever Hans Strategies in Image Classifiers through Generating Counterexamples},
  author={Bender, Sidney and Delzer, Ole and Herrmann, Jan and Marxfeld, Heike Antje and M{\"u}ller, Klaus-Robert and Montavon, Gr{\'e}goire},
  journal={arXiv preprint arXiv:2510.17524},
  year={2025}
}
```

You can reproduce the results from the paper with reproduction_scripts/reproduce_cfkd_results.sh.

If you use DiDAE (https://arxiv.org/pdf/2601.21851) useful consider citing:

```
@article{bender2026visual,
  title={Visual Disentangled Diffusion Autoencoders: Scalable Counterfactual Generation for Foundation Models},
  author={Bender, Sidney and Morik, Marco},
  journal={arXiv preprint arXiv:2601.21851},
  year={2026}
}
```

You can reproduce the results from the paper with reproduction_scripts/reproduce_didae_results.sh.

## Three ways to use PEAL

Every method runs from one config file, so the same experiment can be driven three ways:

1. **Command line.** Each method has a command that takes a config: `peal-cfkd`, `peal-didae`,
   `peal-train-predictor`, `peal-train-generator` and eight more, all listed in the "Commands"
   table of `docs/installation.md` (from a clone, the identically named scripts in the
   repository root, e.g. `python run_cfkd.py --config <config.yaml>`, run the same code). This is also how the papers'
   results are reproduced: `reproduction_scripts/reproduce_{sce,cfkd,didae}_results.sh` are
   sequences of these commands.
2. **Python API.** `import peal` exposes the config models, the factories and the methods
   (`peal.CFKD`, `peal.DiDAE`, `peal.get_generator`, `peal.get_predictor`, ...), loaded lazily so
   the import itself is instant. The commands above are thin wrappers around exactly this:

   ```python
   import peal

   config = peal.load_yaml_config("configs/my_experiments/adaptors/my_cfkd.yaml", peal.CFKDConfig)
   peal.set_random_seed(config.seed)
   student = peal.CFKD(adaptor_config=config).run()  # returns the corrected student
   ```

   `peal.DiDAE` / `peal.DiDAEConfig` work the same way. "Using PEAL components from your own
   code" below lists what each part does and how to plug in your own datasets and models.
3. **Web interface.** Upload an ONNX classifier and a zip with one folder per class, judge the
   counterfactuals in the browser, and download the corrected classifier; no config writing
   needed. See "Web demo" below.

**Tutorial notebooks**

The walkthroughs below also exist as runnable notebooks in `notebooks/`, which derive the config
files for you and plot the resulting counterfactuals: `01_explain_your_classifier_with_sce.ipynb`,
`02_bring_your_own_onnx_predictor_sce.ipynb` and `03_explain_an_onnx_probe_with_didae.ipynb`.
See `notebooks/README.md` for which one to start with.

**Install from PyPI**

```
pip install peal-xai
```

The distribution is called `peal-xai` because `peal` on PyPI is an unrelated
project; the import package is still `peal`. Optional stacks are extras, for
example `pip install "peal-xai[xai,datasets]"` or `pip install "peal-xai[full]"`.
See the installation page of the documentation for the full table.

Note that the wheel does not carry the experiment configs under `configs/`,
which live outside the package. They ship in the source distribution, so to run
the `--config <PEAL_BASE>/configs/...` commands below, clone this repository and
point `$PEAL_BASE` at it. `requirements.txt` pins the exact environment the
papers were produced with and is separate from what `pip install` resolves.

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

An alternative to conda is to work with apptainer by running ```apptainer build python_container.sif deploy/apptainer/python_container.def``` and then run everything inside ```apptainer run --nv python_container.sif```.

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
9) Here, you have to replace model_path with "$PEAL_RUNS/my_data/classifier" and data with
"<PEAL_BASE>/configs/my_experiments/data/my_data.yaml". The label column is chosen by y_selection,
which is NOT a top-level key: it lives inside the `task:` block, so set `task.y_selection` to
"[Label1]". A top-level y_selection is silently ignored, and the run would then train on CelebA's
Smiling attribute instead of your label.
10) Now you can train your predictor with: ```python train_predictor.py --config "<PEAL_BASE>/configs/my_experiments/predictors/my_data_classifier.yaml"```
11) After finishing generator and predictor training, you can copy "configs/sce_experiments/adaptors/celeba_Smiling_natural_sce_cfkd.yaml" 
to "configs/my_experiments/adaptors/my_data_sce_cfkd.yaml".
12) Overwrite data with "<PEAL_BASE>/configs/my_experiments/data/my_data.yaml", student with "$PEAL_RUNS/my_data/classifier/model.cpl", generator with "$PEAL_RUNS/my_data/ddpm/config.yaml", base_dir with "$PEAL_RUNS/my_data/classifier/sce_cfkd", task.y_selection with "[Label1]" (again
inside the `task:` block, not at the top level) and calculate_explainer_stats with "False".
13) Now you can run SCE with: ```python run_cfkd.py --config "<PEAL_BASE>/configs/my_experiments/adaptors/my_data_sce_cfkd.yaml"```
14) Now you can find your most salient counterfactuals under
"$PEAL_RUNS/my_data/classifier/sce_cfkd/0/validation_collages0_1". The "0" is the fine-tuning
iteration: every output of a CFKD run is written under <base_dir>/<iteration>/.
15) You can find secondary counterfactuals under
"$PEAL_RUNS/my_data/classifier/sce_cfkd/0/validation_collages0_0", but it might be possible that they look destroyed if SCE could not find another counterfactual and forced it too much

If you further want to process them, you can load the .npz array
"$PEAL_RUNS/my_data/classifier/sce_cfkd/0/validation_tracked_values.npz".
The originals in this array can be found under the key "x_list" and the counterfactuals under "x_counterfactual_list".
If you additionally want to use CFKD look how you have to configure your adaptor config according to the configs found e.g. in the configs/cfkd_experiments/adaptors.


**How to use a custom image dataset and a custom predictor**


First, you can copy "configs/sce_experiments/adaptors/imagenet_husky_vs_wulf_sce_cfkd.yaml" to "configs/my_experiments/adaptors/my_data_sce_cfkd.yaml".

If you do not wish to use the ImageNet DDPM as generative model one has to copy "configs/sce_experiments/data/imagenet_generator.yaml" to
"configs/my_experiments/data/my_data_generator.yaml" (there is no top-level configs/data/ directory).
Then, the dataset_class in "configs/my_experiments/data/my_data_generator.yaml" has to be removed.
Then, num_samples, input_size and output_size should be adapted to your dataset.
If you do not want to bootstrap your DDPM with Imagenet 256x256 weights you have to remove download_weights.
Now, one has to copy configs/sce_experiments/generators/imagenet_ddpm.yaml to configs/my_experiments/generators/my_data_ddpm.yaml.
In this file one has to replace base_path with "$PEAL_RUNS/my_data/ddpm" and data with "<PEAL_BASE>/configs/my_experiments/data/my_data_generator.yaml".
Now you can train your DDPM generator with:
```python train_generator.py --config "<PEAL_BASE>/configs/my_experiments/generators/my_data_ddpm.yaml"```
Then, in "configs/my_experiments/adaptors/my_data_sce_cfkd.yaml" one has to overwrite generator with
"$PEAL_RUNS/my_data/ddpm/config.yaml" -- train_generator.py always writes <base_path>/config.yaml.

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

# Python does not expand $PEAL_RUNS, so read it from the environment.
RUNS = os.environ.get("PEAL_RUNS", "peal_runs")
os.makedirs(os.path.join(RUNS, "imagenet", "wulf_vs_husky_classifier"), exist_ok=True)
dummy_input = torch.randn(1, 3, 224, 224)  # standard ResNet input
OUTPUT_PATH = os.path.join(RUNS, "imagenet", "wulf_vs_husky_classifier", "model.onnx")
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

Next, one has to copy "configs/sce_experiments/data/imagenet_husky_vs_wulf_resnet18.yaml" to
"configs/my_experiments/data/my_data.yaml". Do NOT copy "imagenet.yaml": that one is
dataset_class Image2ClassDataset with output_size [1000], an ImageFolder-style loader that never
reads the data.csv you just wrote. The husky/wulf file is the Image2MixedDataset variant that does.
In the new file you have to change input_size and normalization to the values your classifier was
trained with, and remove the dataset_class line as in the tutorials above.
Then, in "configs/my_experiments/adaptors/my_data_sce_cfkd.yaml" one has to overwrite data with
"<PEAL_BASE>/configs/my_experiments/data/my_data.yaml", and task.y_selection with "[Label1]"
(inside the `task:` block, not at the top level).

Next, you can run SCE with:
```python run_cfkd.py --config "<PEAL_BASE>/configs/my_experiments/adaptors/my_data_sce_cfkd.yaml"```
Now you can find your counterfactuals under
"$PEAL_RUNS/my_data/classifier/sce_cfkd/0/validation_collages0_1" (and ..._0 for the second
attempt). There is no directory called plain "validation_collages": the code always appends the
validation-run index, and everything sits under <base_dir>/<iteration>/.
If you further want to process them, you can load the .npz array
"$PEAL_RUNS/my_data/classifier/sce_cfkd/0/validation_tracked_values.npz".
The originals in this array can be found under the key "x_list" and the counterfactuals under "x_counterfactual_list".

**How to use DiDAE with a Reference Foundation Model and ONNX probe for automated counterfactual discovery and model correction**

DiDAE (Disentangled Diffusion Autoencoders) ships a reference foundation model, a sparse dictionary (SAE), and a generator (e.g. via HuggingFace or local checkpoint). When you have a new foundation model classifier probe (as an ONNX model or PyTorch `.cpl`), DiDAE automatically discovers latent directions corresponding to spurious correlations and allows you to fine-tune the classifier with zero manual dataset labeling.

The workflow proceeds in 9 automated steps:

1) **Linear Distillation**: The code distills the target classifier probe into the reference foundation model's encoder space using closed-form least-squares regression ($w = (Z^T Z)^{-1} Z^T y$). This step runs in seconds without iterative optimization.

2) **Latent Space Sweep**: Select $N$ sample images (e.g., $N=200$) and compute latent counterfactual shifts across **all** $K$ directions in the sparse dictionary latent space.

3) **Latent-Space Filtering**: Perform pure matrix algebra filtering in latent space to drop any counterfactuals that fail to flip the distilled linear probe's decision ($w^T z_{\text{cf}} \cdot w^T z_{\text{orig}} \ge 0$). This saves compute by skipping expensive image decoding for invalid directions.

4) **Ambient Space Decoding**: Decode the surviving latent counterfactuals into actual pixel-space images using the DAE generator.

5) **Ambient-Space Filtering**: Pass decoded counterfactual images through your original ONNX / PyTorch classifier probe to discard any counterfactuals that fail to flip the full target classifier's prediction.

6) **Direction Ranking**: Sort all sparse dictionary directions by their count of successful ambient-space flips and select the top-$K$ most impactful latent directions.

7) **Visual Cluster Display**: Render contrastive (factual, counterfactual) image collages for each top direction and serve them via the interactive `ClusterTeacher` web interface.

8) **No-Code Human Feedback**: The user opens `http://localhost:8000` in a browser and labels each direction as `"true"` (a semantically valid class modification) or `"false"` (a spurious correlation / Clever Hans strategy).

9) **CFKD Fine-Tuning**: Counterfactuals generated along directions marked as `"false"` are automatically compiled into a corrective training dataset. Deep Feature Reweighting (DFR) is executed to fine-tune the classifier's output layer, removing reliance on the identified spurious features.

---

## Example configuration: `configs/my_experiments/adaptors/my_didae.yaml`

```yaml
adaptor_type                         : "DiDAE"
category                             : "adaptor"
n_samples                            : 200
top_k_directions                     : 10
linesearch_factors                   : ["dynamic"]
decode_batch_size                    : 32
batch_size                           : 200
min_train_samples                    : 800
finetune_iterations                  : 1
continuous_learning                  : "deep_feature_reweighting"
student                              : "$PEAL_RUNS/my_classifier/model.onnx"
teacher                              : "cluster@8000"
generator                            : "$PEAL_RUNS/reference_dae/config.yaml"
sparse_dictionary                    : "$PEAL_RUNS/reference_dae/BatchTopKSAE/config.yaml"
data                                 : "<PEAL_BASE>/configs/my_experiments/data/my_data.yaml"
base_dir                             : "$PEAL_RUNS/my_classifier/didae"
task:
  output_channels                    : 2
  y_selection                        : ["TargetLabel"]
```

## Executing DiDAE

```bash
python run_didae.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/only_sparse_numbers1kx098_didae.yaml"
```

Once executed:
- Open `http://localhost:8000` in your web browser.
- Review the collage pairs for each of the top latent directions.
- Label directions as `"true"` or `"false"`.
- The fine-tuned, robustified model will be saved to `$PEAL_RUNS/my_classifier/didae/model.cpl`.

---

**Walkthrough: explain an ONNX classifier or foundation-model probe end to end**

The section above describes what DiDAE does. This one is the concrete recipe for the case where all
you have is a model in ONNX format and a folder of images. Nothing is trained on your model: the
reference generator and its dictionary are reused as they are, and your model is only ever called
forward. That is what makes ONNX enough.

**1. Export your model.** PEAL feeds the ONNX graph exactly the tensor the dataset produces, with
no normalization of its own (see `peal/architectures/predictors.py`), so the graph has to accept
what your data config emits and return one logit vector per image:

```python
import torch
torch.onnx.export(
    my_model.eval(),                       # anything callable on a float tensor
    torch.randn(1, 3, 224, 224),           # the input shape your data config will produce
    "$PEAL_RUNS/my_model/model.onnx",
    input_names=["input"], output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},   # batching matters
)
```
A worked example that wraps a torchvision model and slices two ImageNet classes out of its head is
`tools/convert_imagenet_resnet18_to_binary_onnx.py`.

**2. Put your images where PEAL can read them.** Same layout as the SCE walkthrough above: a
`<PEAL_DATA>/my_data/data.csv` with the header `ImagePath,MyLabel` and the images beside it. Then
copy a data config, e.g. `configs/didae_experiments/data/imagenet_koala_vs_wombat.yaml`, to
`configs/my_experiments/data/my_data.yaml` and set `dataset_path`, `input_size` and — the one that
bites — `normalization` to whatever your ONNX graph expects. Setting it to `null` hands the graph
raw `[0, 1]` images.

**3. Pick a reference generator and dictionary.** These are the two pretrained components DiDAE
composes; you do not train either. For natural images use the representation autoencoder over
OpenAI CLIP ViT-L/14 together with the public 6144-atom MSAE:

```yaml
generator         : "<PEAL_BASE>/configs/web_demo/imagenet_rae_clip_hf.yaml"
sparse_dictionary : "<PEAL_BASE>/configs/didae_experiments/sparse_dictionaries/msae_decomposition.yaml"
```
Two things that config needs before it will load. It resolves its weights from
`$PEAL_RAE_WEIGHTS`, a `hf://<repo>` or a local folder holding `decoder.pt`, `stats.pt` and
`stage2_ema.pt`; see `tools/export_rae_weights.py` for producing one from a training run. And
because it is a `RAEDiffusionAutoencoder`, it needs the RAEv2 fork, which ships separately
under CC BY-NC 4.0: `pip install "peal-xai[rae]"` or `python tools/install_rae.py` (see
"Optional: the RAE generators" above). The preset under `$PEAL_RUNS/imagenet/rae_clip/` that the ImageNet
experiments used is NOT portable: it pins an absolute checkpoint path on the machine it was
trained on.
The generator's encoder and your model's encoder need not be the same network — the ImageNet
experiment ranks a DINOv3 probe against a dictionary that lives in CLIP space.

**4. Write the adaptor config**, `configs/my_experiments/adaptors/my_data_didae.yaml`. Copy
`configs/didae_experiments/adaptors/imagenet_koala_vs_wombat_dinov3_ep43s100_msae.yaml` and change
every one of these keys. The first three are the obvious ones; `teacher` and `test_data` are
easy to miss and both point at files that exist only on our machine. `DiDAE.__init__` builds the
teacher and resolves `test_data` before it looks at `finetune_iterations`, so leaving them alone
makes the run die on construction even with `finetune_iterations: 0`:

```yaml
student                              : "$PEAL_RUNS/my_model/model.onnx"
data                                 : "<PEAL_BASE>/configs/my_experiments/data/my_data.yaml"
test_data                            : "<PEAL_BASE>/configs/my_experiments/data/my_data.yaml"
teacher                              : "human@8000"   # or your own unpoisoned .cpl
base_dir                             : "$PEAL_RUNS/my_model/didae"
task:
  output_channels                    : 2
  y_selection                        : ["MyLabel"]
```
Useful knobs: `top_k_directions` caps how many directions are decoded and shown, `n_samples` how
many images the sweep runs over, and `sweep_atom_subset` restricts the sweep to named atoms when
you already suspect one.

**5. Run it.**

```bash
python -W ignore run_didae.py --config "<PEAL_BASE>/configs/my_experiments/adaptors/my_data_didae.yaml"
```

**6. Read the result.** Under `$PEAL_RUNS/my_model/didae/`:

| file | what it holds |
|------|---------------|
| `sweep_results.pt` | every direction with its latent, ambient and verified flip counts |
| `direction_collages/rank<NNN>_dir<idx>_<name>/` | before/after pairs per direction, best first |
| `successful_flips/` | the counterfactuals that flipped your model, grouped by direction |
| `matches.txt` | which dictionary atom matches which ground-truth attribute, when labels exist |
| `distilled_probe/` | the linear probe fitted to your model in the encoder space |

The ranking is by *verified* flips: a direction counts only when the decoded image flips your model
and a re-encoding confirms the intended concept actually moved. Check `distilled_probe` first — if
the probe does not agree with your model, the ranking is describing the probe, not your model.

**7. Repair, if you want it.** The teacher decides which directions are spurious. `teacher:
"cluster@8000"` serves the collages at `http://localhost:8000` for you to label by hand; a path to
an unpoisoned `.cpl` model uses that model as an oracle instead. Counterfactuals along directions
marked `false` then become a corrective training set.

**One limitation to plan around:** an ONNX student can usually be fine-tuned as well as explained.
PEAL converts the graph into a trainable torch module with `onnx2torch`, so the repair step gets the
module `ModelTrainer` needs. The conversion is not universal: a graph containing an operator
`onnx2torch` does not implement falls back to an `onnxruntime` closure, which can be explained and
ranked but NOT fine-tuned, and a warning names the operator. The DINOv3 probes used in the ImageNet
experiments are such a case; their exported graphs contain a `Size` node that fails to convert.
So check the warning on your own model. If it does fall back, either stop after the explanation with
`finetune_iterations: 0`, or point `student` at the same model as a PyTorch `.cpl` for the repair and
keep the ONNX file for the sweep. With a `.cpl` student,
`continuous_learning: "deep_feature_reweighting"` retrains only the last layer, which is usually
what you want for a probe. Setting `PEAL_ONNX_CONVERT=0` forces the closure if you want the
inference-only path deliberately.

---

## Using PEAL components from your own code

You can utilize different top-level scripts to use the library:

```python generate_dataset.py --config "<PATH_TO_CONFIG>"```
Generates a synthetic dataset like the squares dataset based on a data config file.
Another option is to augment a already downloaded dataset like CelebA with a controlled confounder.
Existing data config files can be found in configs/sce_experiments/data.
The generated dataset will appear in $PEAL_DATA.
For the replication of the experiments on synthetic / augmented datasets from the paper one first needs to generate the articial datasets.
A full, documented overview over the parameters that can be set by datasets so far can be found in ```peal.data.interfaces.DataConfig```.
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
Existing generator config files can be found in configs/sce_experiments/generators.
In order to e.g. use counterfactual explanations one needs to train a corresponding generator.
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
To implement a new explainer ```MyExplainer``` create a subclass of ```peal.explainers.interfaces.ExplainerInterface``` inside ```peal.explainers```.
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

## Example workflow: the CFKD adaptor

Here we introduce a code snippet that does the same as "run_cfkd.py" does in a no-code CLI manner.
Assuming you have a predictor ```my_classifier``` and a dataset ```my_dataset``` and a peal.adaptors.counterfactual_knowledge_distillation.CFKDConfig ```adaptor_config``` configuring your CFKD run.

```
from peal.adaptors.counterfactual_knowledge_distillation import CFKD

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

2) A generative model is trained based on configs/sce_experiments/generators/<your generator>.yaml and saved under peal_runs/run1/generator .

3) A i'th round of counterfactuals is calculated and the explanation collages under peal_runs/run1/i/collages

4) A web interface is started under localhost:8000 that receives feedback from the user and saves it

5) The counterfactuals are saved with their feedback-adapted label under peal_runs/run1/i/dataset

6) The classifier is finetuned based on configs/cfkd_experiments/training/img_finetuning.yaml and saved under peal_runs/run1/i/finetuned_model/model.cpl

7) If i smaller then the maximum number of finetune iterations go back to 3.


## Structure of the project

peal/explainers - the different explainers (e.g. counterfactual explanations, layer-wise relevance explanations...)

peal/teachers - the different teachers (e.g. human teacher, oracle teacher, segmentation mask teacher...)

peal/adaptors - the different model adaptors, that are able to refine a model (e.g. counterfactual knowledge distillation, projective class artifact compensation, ...)

peal/architectures - architecture components used for the available predictors

peal/training - everything that is needed for training and finetuning predictors

peal/data - datasets, dataloaders and data generators, that e.g. allow creating controlled confounder dataset based on copyright tag

peal/generators - the different generative models that can be used to generate counterfactuals

peal/dependencies - integration of related work that does not provide a library

configs - generic config files, that can be either directly used, extended, adapted or exchanged

reproduction_scripts - the shell scripts that reproduce the SCE, CFKD and DiDAE paper results, plus the multi-GPU launchers of the ImageNet / CelebA generator trainings

tools - stand-alone helper scripts (RAEv2 install, weight export, ONNX export of a torchvision model, ImageNet pair extraction, SpRAy labelling, speed benchmark, finetuning grid search)

deploy - the Docker / compose files of the web demo (spark/) and the apptainer recipe of the cluster environment (apptainer/)

notebooks - the notebooks that walk you step by step how to use the library e.g. to reproduce results of the papers

templates - original html templates for the feedback webapp

tests - unit tests to ensure the components work properly on mock data

docs - the Sphinx sources of the project documentation (user guide plus the API
reference generated from the docstrings). Build it with
```pip install -r docs/requirements.txt && sphinx-build -b html docs docs/_build/html```
and open docs/_build/html/index.html; .github/workflows/docs.yml publishes the same
build to GitHub Pages.

**Optional: the RAE generators**

The representation-autoencoder generators (`RAEDiffusionAutoencoder`, used for the
ImageNet and CelebA RAE pipelines) are built on a modified RAEv2, which is licensed CC BY-NC
4.0 (non-commercial) and therefore ships as its own package, `peal-xai-rae`
(`packages/peal-xai-rae` in this repository), never inside `peal-xai`. If you want them:

```
pip install "peal-xai[rae]"             # from PyPI
python tools/install_rae.py             # from a clone: shows the licence, then installs it
```

From a clone PEAL also finds the in-repository copy without any install. The
ImageNet weights (6.6 GB) are downloaded from Hugging Face on first use.

Nothing else in PEAL needs it. The ImageNet and CelebA diffusion autoencoders and
PEAL's own edit-friendly DDPM and DDIM inversion run without RAEv2, so a
published diffusion-autoencoder checkpoint can be loaded, inverted and edited on
a plain install.

**Installing PEAL itself**

```
pip install -e .                  # from a clone
pip install -e ".[onnx,web]"      # ONNX classifiers and the web demo
pip install -e ".[pathldm]"       # PathLDM generators
pip install -e ".[stablediffusion]"
```

The wheel carries `peal/` including the vendored `peal/dependencies/`, minus the
two components above. It does **not** carry `configs/`: the 800 experiment
configs sit outside the package and are referenced as `<PEAL_BASE>/configs/...`.
From a clone that resolves by itself. From an installed PEAL, point `$PEAL_BASE`
at a source tree:

```
export PEAL_BASE=/path/to/pytorch_explain_and_adapt_library
```

## Licence and citation

PEAL is released under the GNU Lesser General Public License v3 or later. The
notice is in `LICENSE.txt` and the full texts are in `COPYING.LESSER` and
`COPYING`.

**One part is non-commercial.** The RAE generators (`RAEDiffusionAutoencoder`,
the ImageNet and CelebA representation autoencoders, and the published RAE
weights) are built on RAEv2, which its authors license CC BY-NC 4.0. The modified
RAEv2 they need is in this repository under `packages/peal-xai-rae`, keeps its
CC BY-NC 4.0 licence, and ships as the separate package `peal-xai-rae`; the
`peal-xai` package contains none of it. Using those generators is non-commercial
regardless of PEAL's own licence. See `LICENSING.md`.

Every other generator is free of that restriction, including the diffusion
autoencoder behind the Square, CelebA and Camelyon17 results. Generators are
selected by name in a config, so a commercial user swaps the generator rather
than forking the library. **`LICENSING.md` is the full map** of what is usable
commercially and what is not.

Everything under `peal/dependencies/` is third-party research code vendored so
that PEAL's baselines run unmodified, and it keeps its own licence. Each folder
records its upstream commit and how this copy differs, and
`THIRD_PARTY_NOTICES.md` lists every component with its licence. Read both files
before redistributing a clone: two vendored components still have unresolved
provenance.

No model weights and no datasets ship with PEAL. Cite the paper for the method
you used; `CITATION.cff` carries the machine-readable entries. To cite the
software itself, use the Zenodo record of the release you used:

```bibtex
@misc{bender2026peal,
  title        = {{PEAL}: PyTorch Explain-and-Adapt Library},
  author       = {Bender, Sidney and Kunz, Benedikt and Zeid, Ahmed and Delzer, Ole},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {0.1.0},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://github.com/Explainable-AI-Berlin/pytorch_explain_and_adapt_library}
}
```

<!-- TODO(release): replace 10.5281/zenodo.XXXXXXX above and in CITATION.cff with the
     DOI Zenodo mints for the v0.1.0 release, and add the badge:
     [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

## Contribution guidelines

If you want to contribute to the project, please follow the following guidelines:

1) Use the black formatter with 88 columns for python files. The options are in
pyproject.toml and the pre-commit hook enforces them; run
```pip install pre-commit && pre-commit install``` once per clone so every commit is
formatted, and ```black .``` to format by hand. peal/dependencies (vendored third-party
code) is excluded on purpose.

2) Avoid code redundancy as much as possible

3) Write unit tests for every new component that have high code coverage

4) Write sphinx parsable documentation for every new component in the NumPy documentation style.
Every module, class and public function has a docstring; the API reference under docs/
is generated from them, so a missing or wrong docstring shows up on the site.

5) Extend existing code instead of changing it!!!

6) Compositionality is key! Try to make the components as independent as possible and make sure that the whole pipeline still works if you replace a component with another one.

7) Always use seeds for every experiment to make sure the results are exactly reproducible

8) Log all important information in the log files to make experiments comparable

9) No classes longer than 500 lines of code and no methods longer than 100 lines of code

10) Use design patterns like the factory or visitor pattern to make the code more readable and maintainable

11) Benchmark your code that gpu utilization is high enough instead of being to busy with loading data or moving it between cpu and gpu

## Web demo (upload an ONNX classifier, get a corrected one back)

`peal/web` is a small FastAPI app around `run_didae.py`: users drop an ONNX
image classifier and a zip with one folder per class, fill in what the model
expects (the two classes to analyse, their output indices, the input
normalization and size), judge the rendered counterfactuals of the top
directions as true or spurious, and download the finetuned classifier as ONNX.
The generator and dictionary weights come from Hugging Face
(`tools/export_rae_weights.py` publishes the RAE; MSAE downloads itself).

    pip install -r requirements.txt          # the base install, needed first
    pip install -r requirements-web.txt      # fastapi, uvicorn, onnx, onnx2torch
    python tools/install_rae.py              # the generator is a RAEDiffusionAutoencoder (CC BY-NC)
    export PEAL_RAE_WEIGHTS=hf://sidney1505/peal-rae-clip-imagenet@v0.1.0
    uvicorn peal.web.app:app --host 0.0.0.0 --port 8080

The published ImageNet weights live at <https://huggingface.co/sidney1505/peal-rae-clip-imagenet>
(CC BY-NC 4.0, tag `v0.1.0` is the revision used in the paper). You can also point
`$PEAL_RAE_WEIGHTS` at a local folder holding `decoder.pt`, `stats.pt` and `stage2_ema.pt`, for
example one produced by `tools/export_rae_weights.py` from your own RAE run. The config builder rejects an unset `CHANGE_ME` value with a clear message rather
than failing later. Note also that `requirements-web.txt` asks for the CPU `onnxruntime` while
`requirements.txt` pins `onnxruntime_gpu`; install the web extras in their own environment rather
than on top of a GPU install.

`deploy/spark/` has the Dockerfile, compose file and notes for running it on
a DGX Spark. ONNX classifiers are converted to trainable torch modules with
onnx2torch (`peal/architectures/onnx_predictor.py`), so CFKD can finetune them;
graphs onnx2torch cannot convert fall back to onnxruntime and can only be
analysed.
