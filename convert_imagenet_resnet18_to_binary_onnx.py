import torch
import os
import torchvision
import argparse


class BinaryImageNetModel(torch.nn.Module):
    def __init__(self, class1, class2):
        super(BinaryImageNetModel, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.class1 = class1
        self.class2 = class2

    def forward(self, x):
        logits_full = self.model(x)
        return logits_full[:, [self.class1, self.class2]]


def convert_to_binary_onnx(class1=248, class2=269, name="husky_vs_wulf"):
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
