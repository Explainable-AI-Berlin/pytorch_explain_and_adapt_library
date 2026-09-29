"""Export a two-class slice of torchvision's ImageNet ResNet-18 as ONNX.

Worked example, referenced from the README, of bringing an external model
into PEAL: ``BinaryImageNetModel`` wraps the pretrained ResNet-18 and returns
only the logits of two ImageNet classes, and the wrapper is exported with a
dynamic batch axis so that the ``onnx`` architecture
(``peal/architectures/onnx_predictor.py``) can load it as a predictor.

Invocation::

    python tools/convert_imagenet_resnet18_to_binary_onnx.py \\
        [--class1 248] [--class2 269] [--name husky_vs_wulf]

Writes ``$PEAL_RUNS/imagenet/<name>_classifier/model.onnx`` and creates the
directory; ``PEAL_RUNS`` must be set. The default pair is Eskimo dog (248)
as class 0 vs. white wolf (269) as class 1. The predictor configs in
``configs/cfkd_fastdime_hq/predictors/imagenet_*_classifier.yaml`` record the
invocation that produced each ONNX file.
"""

import torch
import os
import torchvision
import argparse


class BinaryImageNetModel(torch.nn.Module):
    """Pretrained ResNet-18 restricted to the logits of two ImageNet classes.

    Parameters
    ----------
    class1 : int
        ImageNet index that becomes output channel 0.
    class2 : int
        ImageNet index that becomes output channel 1.
    """

    def __init__(self, class1, class2):
        super(BinaryImageNetModel, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.class1 = class1
        self.class2 = class2

    def forward(self, x):
        """Compute the two selected logits.

        Parameters
        ----------
        x : torch.Tensor
            Image batch of shape ``(N, 3, 224, 224)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(N, 2)``.
        """
        logits_full = self.model(x)
        return logits_full[:, [self.class1, self.class2]]


def convert_to_binary_onnx(class1=248, class2=269, name="husky_vs_wulf"):
    """Build the two-class wrapper and export it to ``$PEAL_RUNS``.

    Parameters
    ----------
    class1 : int, optional
        ImageNet index for output channel 0.
    class2 : int, optional
        ImageNet index for output channel 1.
    name : str, optional
        Run name; the file lands in ``$PEAL_RUNS/imagenet/<name>_classifier``.

    Returns
    -------
    None
    """
    binary_classifier = BinaryImageNetModel(class1, class2)
    binary_classifier.eval()
    peal_runs = os.environ.get("PEAL_RUNS")

    if not os.path.exists(peal_runs + "/imagenet"):
        os.makedirs(peal_runs + "/imagenet")

    if not os.path.exists(peal_runs + "/imagenet/" + name + "_classifier"):
        os.makedirs(peal_runs + "/imagenet/" + name + "_classifier")

    dummy_input = torch.randn(1, 3, 224, 224)  # standard ResNet input
    OUTPUT_PATH = peal_runs + "/imagenet/" + name + "_classifier/model.onnx"
    torch.onnx.export(
        binary_classifier,
        dummy_input,
        OUTPUT_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11,
    )
    print("onnx model saved at: " + OUTPUT_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some class arguments.")

    parser.add_argument(
        "--class1", type=int, default=248, help="Value for class1 (default: 248)"
    )
    parser.add_argument(
        "--class2", type=int, default=269, help="Value for class2 (default: 269)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="husky_vs_wulf",
        help="Name value (default: wulf_vs_husky)",
    )
    args = parser.parse_args()
    convert_to_binary_onnx(class1=args.class1, class2=args.class2, name=args.name)
