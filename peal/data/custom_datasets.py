import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.transforms import ToTensor

from peal.data.dataset_generators import SquareDatasetGenerator
from peal.data.datasets import (
    SymbolicDataset,
    Image2ClassDataset,
    DataConfig,
    Image2MixedDataset,
)
from peal.generators.interfaces import Generator


class CircleDataset(SymbolicDataset):
    def __init__(
        self, data_dir, mode, config, transform=ToTensor(), task_config=None, **kwargs
    ):
        super(CircleDataset, self).__init__(
            data_dir=data_dir,
            mode=mode,
            config=config,
            transform=transform,
            task_config=task_config,
            **kwargs,
        )

    @staticmethod
    def circle_fid(samples):
        radius = 1
        return (((samples.pow(2)).sum(dim=-1) - radius).pow(0.5)).mean()

    @staticmethod
    def angle_cdf(samples):
        scores = abs(samples[:, 1] / samples[:, 0])

        first_quad_mask = (samples[:, 0] > 0) & (samples[:, 1] > 0)
        second_quad_mask = (samples[:, 0] < 0) & (samples[:, 1] > 0)
        third_quad_mask = (samples[:, 0] < 0) & (samples[:, 1] < 0)
        fourth_quad_mask = (samples[:, 0] > 0) & (samples[:, 1] < 0)
        theta_1 = torch.atan(scores) * first_quad_mask
        theta_1 = theta_1[theta_1 != 0]
        theta_2 = (torch.pi - torch.atan(scores)) * second_quad_mask
        theta_2 = theta_2[theta_2 != 0]
        theta_3 = (torch.pi + torch.atan(scores)) * third_quad_mask
        theta_3 = theta_3[theta_3 != 0]
        theta_4 = (2 * torch.pi - torch.atan(scores)) * fourth_quad_mask
        theta_4 = theta_4[theta_4 != 0]
        thetas, indices = torch.cat([theta_1, theta_2, theta_3, theta_4]).sort(dim=-1)

        return thetas

    def circle_ks(self, samples, true_data):
        true_thetas = CircleDataset.angle_cdf(true_data)
        sample_thetas = CircleDataset.angle_cdf(samples)

        ecdf = torch.arange(self.config["num_samples"]) / self.config["num_samples"]
        true_cdf = (sample_thetas[:, None] >= true_thetas[None, :]).sum(-1) / len(
            true_data
        )
        return torch.max(torch.abs((true_cdf - ecdf)))

    # def track_generator_performance(self, generator: Generator, batch_size=1):
    #    samples = generator.sample_x(batch_size).detach()

    #    ks = self.circle_ks(samples, self.true_data)
    #    fid = CircleDataset.circle_fid(samples)

    #    harmonic_mean = 1 / (1 / fid + 1 / ks)

    #    return {
    #        'KS': ks,
    #        'FID': fid,
    #        'harmonic_mean_fid_ks': harmonic_mean

    #    }

    def generate_contrastive_collage(
        self,
        x_list,
        x_counterfactual_list,
        y_target_list,
        y_source_list,
        target_confidence_goal,
        base_path,
        start_idx,
        classifier=None,
        dataloader=None,
        **kwargs,
    ):
        collage_paths = []
        # base_path = Path(base_path).parent
        Path(base_path).mkdir(parents=True, exist_ok=True)

        data = torch.zeros([len(self.data), len(self.attributes)], dtype=torch.float16)
        for idx, key in enumerate(self.data):
            data[idx] = self.data[key]

        input_idx = [
            idx
            for idx, element in enumerate(self.attributes)
            if element not in self.config.confounding_factors
        ]

        # plotting counterfactuals
        plt.figure()
        plt.scatter(data[:, input_idx[0]], data[:, input_idx[1]], color="lightgray")

        counterfactual_path = os.path.join(base_path, str(start_idx))
        Path(counterfactual_path).mkdir(parents=True, exist_ok=True)
        for i, point in enumerate(x_counterfactual_list):
            plt.scatter(x_list[i][0], x_list[i][1], color="green")
            plt.scatter(point[0], point[1], color="red", label="end")
            import pdb

            pdb.set_trace()
            plt.arrow(
                x_list[i][0],
                x_list[i][1],
                point[0] - x_list[i][0],
                point[1] - x_list[i][1],
                head_width=0.05,
                head_length=0.05,
                fc="blue",
                ec="blue",
            )
        # flip_rate = np.mean((classifier(torch.stack(x_counterfactual_list, dim=0)).softmax(dim=-1).argmax(dim=-1) != classifier(torch.stack(x_list, dim=0)).softmax(dim=-1).argmax(dim=-1)).numpy())
        plt.show()
        plt.savefig(counterfactual_path + "/counterfactuals.png")
        collage_paths.append(counterfactual_path)

        # plotting the train dataset
        data_path = os.path.join(base_path, "data.png")
        collage_paths.append(data_path)
        plt.figure()
        # if np.random.rand() < 0.5:
        #    import pdb; pdb.set_trace()
        # try:
        # train_path = Path(base_path).parent
        if not dataloader is None:
            xs = []
            ys = []
            for i in range(100):
                x, y = dataloader.sample()
                # if x.shape[0] == 100:
                xs.append(x)
                ys.append(y)
            xs = torch.stack([tensor for i in range(len(xs)) for tensor in xs[i]])
            ys = torch.stack([tensor for i in range(len(ys)) for tensor in ys[i]])
            # df = pd.read_csv(base_path/'train_dataset.csv').to_numpy()
            plt.scatter(data[:, input_idx[0]], data[:, input_idx[1]], color="lightgray")
            # plt.scatter(df[:, 0], df[:, 1], c=np.where(df[:, -1] == 0, 'green', 'red'))
            plt.scatter(xs[:, 0], xs[:, 1], c=np.where(ys == 0, "green", "red"))
            plt.show()
            plt.savefig(data_path)
        # except FileNotFoundError:
        #    pass

        # plotting gradient scalar field
        grad_path = os.path.join(base_path, "gradient_field.png")
        collage_paths.append(grad_path)
        fig, axs = plt.subplots(1, 2, figsize=(20, 7))
        xx1, xx2 = np.meshgrid(
            *[
                np.linspace(
                    float(data[:, [idx]].min() - 0.5),
                    float(data[:, [idx]].max() + 0.5),
                    20,
                )
                for idx in input_idx
            ]
        )
        grid = torch.from_numpy(np.array([xx1.flatten(), xx2.flatten()]).T).to(
            torch.float32
        )
        grid.requires_grad = True
        for i in range(2):
            input_data = torch.nn.Parameter(grid.clone())
            logits = classifier(input_data)
            logits[:, i].sum().backward()
            input_gradients = input_data.grad
            axs[i].quiver(
                input_data[:, 0].detach(),
                input_data[:, 1].detach(),
                input_gradients[:, 0],
                input_gradients[:, 1],
            )
            input_data.grad.zero_()
            axs[i].set_title(f"Class:{i}")
        plt.show()
        plt.savefig(grad_path)

        # plotting contours

        contour_path = os.path.join(base_path, "contours.png")

        xx1, xx2 = np.meshgrid(
            *[
                np.linspace(
                    float(data.data[:, [input_idx]].min() - 0.5),
                    float(data.data[:, [input_idx]].max() + 0.5),
                    200,
                )
                for idx in input_idx
            ]
        )

        grid = torch.from_numpy(np.array([xx1.flatten(), xx2.flatten()]).T).to(
            torch.float32
        )
        contour_logits = classifier(grid).detach()
        contour_logits_diff = (contour_logits[:, 1] - contour_logits[:, 0]).reshape(
            xx1.shape
        )
        plt.figure()
        plt.scatter(data[:, input_idx[0]], data[:, input_idx[1]], color="lightgray")
        plt.contour(
            xx1,
            xx2,
            contour_logits_diff,
            levels=torch.linspace(
                contour_logits_diff.min(), contour_logits_diff.max(), 10
            ).tolist(),
            lcmap="coolwarm",
        )
        plt.contour(
            xx1, xx2, contour_logits_diff, levels=[0], colors="red", label="level 0"
        )
        plt.text(1.0, 1.0, "level 0: red line")
        # fig, axs = plt.subplots(1, 2, figsize=(20, 7))

        # for i in range(2):
        #    axs[i].scatter(data[:, 0], data[:, 1], c=np.where(data[:, -1] == 0, 'lightcyan', 'lightgray')[0])
        #    z1 = contour[:, i].reshape(xx1.shape)
        #    axs[i].contour(xx1, xx2, z1, levels=torch.linspace(contour[:, i].min(), contour[:, i].max(), 10).tolist(),
        #                   lcmap='coolwarm')
        #    axs[i].contour(xx1, xx2, z1, levels=[0], colors='red')
        #    axs[i].text(1.0, 1.0, 'level 0: red line')
        #    axs[i].set_title(f'Class:{i}')
        plt.show()
        plt.savefig(contour_path)

        return x_list, collage_paths

    def _initialize_performance_metrics(self):
        data = torch.zeros([len(self.data), len(self.attributes)], dtype=torch.float32)
        for idx, key in enumerate(self.data):
            data[idx] = self.data[key]
        self.true_data = data
        self.true_thetas = CircleDataset.angle_cdf(data)

    def distribution_distance(self, x_list_collection):
        fid_like = []
        for i in range(len(x_list_collection)):
            counterfactuals = torch.stack(x_list_collection[i], dim=0)
            manifold_distance = self.circle_fid(counterfactuals)
            sample_thetas = CircleDataset.angle_cdf(counterfactuals)
            ecdf = torch.arange(len(counterfactuals)) / len(counterfactuals)
            true_cdf = (sample_thetas[:, None] >= self.true_thetas[None, :]).sum(
                -1
            ) / len(self.true_data)
            # how far is the cdf from uniform distribution
            uniformity = torch.max(torch.abs((true_cdf - ecdf)))
            fid_like.append(1 / (1 / manifold_distance + 1 / uniformity))

    def pair_wise_distance(self, x1_collection, x2_collection):
        distances = []
        for i in range(len(x1_collection)):
            distance = (
                (
                    torch.stack(x1_collection[i], dim=0)
                    - torch.stack(x2_collection[i], dim=0)
                )
                .pow(2)
                .sum()
                .pow(0.5)
            )
            distances.append(distance.item())

        return np.mean(distances)

    def variance(self, x_list_collection):
        variances = []
        for i in range(len(x_list_collection[0])):
            variance = torch.mean(
                torch.var(
                    torch.stack(
                        [
                            x_list_collection[j][i]
                            for j in range(len(x_list_collection))
                        ],
                        dim=0,
                    ),
                    dim=0,
                )
            )
            variances.append(variance)

        return np.mean(variances)

    def flip_rate(self, y_confidence_list):
        flip_rates = []
        for i in range(len(y_confidence_list)):
            flip_rate = torch.mean((torch.stack(y_confidence_list[i]) > 0.5).float())
            flip_rates.append(flip_rate)

        return np.mean(flip_rates)


class MnistDataset(Image2ClassDataset):
    def __init__(self, config: DataConfig, **kwargs):
        if not os.path.exists(config.dataset_path):
            mnist_dataset_train = torchvision.datasets.MNIST(
                root=config.dataset_path + "_train_raw",
                train=True,
                download=True,
                transform=None,
            )
            img_dir = os.path.join(config.dataset_path, "imgs")
            idxs = np.zeros([10])
            Path(img_dir).mkdir(parents=True, exist_ok=True)
            for i in range(len(mnist_dataset_train)):
                img, label = mnist_dataset_train[i]
                if not os.path.exists(f"{img_dir}/{label}"):
                    os.makedirs(f"{img_dir}/{label}")

                img.save(f"{img_dir}/{label}/{idxs[label]}.png")
                idxs[label] += 1

            mnist_dataset_val = torchvision.datasets.MNIST(
                root=config.dataset_path + "_val_raw",
                train=False,
                download=True,
                transform=None,
            )
            for i in range(len(mnist_dataset_val)):
                img, label = mnist_dataset_val[i]
                img.save(f"{img_dir}/{label}/{idxs[label]}.png")
                idxs[label] += 1

        super(MnistDataset, self).__init__(config=config, **kwargs)


def plot_latents_with_arrows(original_latents, counterfactual_latents, filename):
    fig, ax = plt.subplots()

    # Convert to numpy arrays for easy manipulation
    original_latents = np.array(original_latents)
    counterfactual_latents = np.array(counterfactual_latents)

    # Plotting the original latents and counterfactuals
    for i, (orig, cf) in enumerate(zip(original_latents, counterfactual_latents)):
        if orig[0] < 0.5:
            # Points with x < 0.5 are dark blue, counterfactuals are light blue
            ax.scatter(orig[0], orig[1], color='darkblue', label='Original (x < 0.5)' if i == 0 else "")
            ax.scatter(cf[0], cf[1], color='lightblue', label='Counterfactual (x < 0.5)' if i == 0 else "")
        else:
            # Points with x >= 0.5 are red, counterfactuals are light red
            ax.scatter(orig[0], orig[1], color='red', label='Original (x >= 0.5)' if i == 0 else "")
            ax.scatter(cf[0], cf[1], color='lightcoral', label='Counterfactual (x >= 0.5)' if i == 0 else "")

        # Draw arrow between original and counterfactual points
        ax.annotate('', xy=(cf[0], cf[1]), xytext=(orig[0], orig[1]),
                    arrowprops=dict(facecolor='green', edgecolor='green', arrowstyle='->'))

    # Setting limits for the plot (0 to 1 for both axes)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Add grey dotted lines at x == 0.5 and y == 0.5
    ax.axvline(x=0.5, color='grey', linestyle='--', linewidth=1)
    ax.axhline(y=0.5, color='grey', linestyle='--', linewidth=1)

    # Axis labels
    ax.set_xlabel('Foreground Intensity')
    ax.set_ylabel('Background Intensity')

    # Create a legend with the correct markers
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    # Show the plot
    plt.grid(True)
    plt.savefig(filename)


def test():
    # Example usage
    original_latents = [(0.2, 0.4), (0.6, 0.7), (0.4, 0.1), (0.8, 0.9)]
    counterfactual_latents = [(0.3, 0.5), (0.7, 0.6), (0.35, 0.15), (0.75, 0.85)]

    plot_latents_with_arrows(original_latents, counterfactual_latents, "bla.png")

class SquareDataset(Image2MixedDataset):
    def __init__(self, config: DataConfig, **kwargs):
        if not os.path.exists(config.dataset_path):
            cdg = SquareDatasetGenerator(data_config=config)
            cdg.generate_dataset()

        super(SquareDataset, self).__init__(config=config, **kwargs)

    def global_counterfactual_visualization(self, counterfactuals, filename, num_samples):
        original_latents = []
        counterfactual_latents = []
        hints_enabled_buffer = self.hints_enabled
        self.hints_enabled = True
        for idx in range(num_samples):
            x, (y, hint) = self[idx]
            original_latents.append(
                [self.check_foreground(x, hint), self.check_background(x, hint)]
            )
            counterfactual_latents.append(
                [
                    self.check_foreground(counterfactuals[idx], hint),
                    self.check_background(counterfactuals[idx], hint),
                ]
            )

        self.hints_enabled = hints_enabled_buffer
        plot_latents_with_arrows(original_latents, counterfactual_latents, filename)

    def check_foreground(self, x, hint):
        intensity_foreground = torch.sum(hint * x) / torch.sum(hint)
        return intensity_foreground

    def check_background(self, x, hint):
        intensity_background = torch.sum((1 - hint) * x) / torch.sum(1 - hint)
        return intensity_background

    def generate_contrastive_collage(
        self,
        x_list: list,
        x_counterfactual_list: list,
        y_target_list: list,
        y_source_list: list,
        y_list: list,
        y_target_start_confidence_list: list,
        y_target_end_confidence_list: list,
        base_path: str,
        idx_to_info=None,
        **kwargs: dict,
    ):
        if idx_to_info is None:

            def idx_to_info(x, x_counterfactual, hint):
                s = (
                    "Foreground: "
                    + str(round(float(self.check_foreground(x, hint)), 3))
                    + " -> "
                )
                s += (
                    str(round(float(self.check_foreground(x_counterfactual, hint)), 3))
                    + ", "
                )
                s += (
                    "Background: "
                    + str(round(float(self.check_background(x, hint)), 3))
                    + " -> "
                )
                s += str(round(float(self.check_background(x_counterfactual, hint)), 3))
                return s

        return super().generate_contrastive_collage(
            x_list=x_list,
            x_counterfactual_list=x_counterfactual_list,
            y_target_list=y_target_list,
            y_source_list=y_source_list,
            y_list=y_list,
            y_target_start_confidence_list=y_target_start_confidence_list,
            y_target_end_confidence_list=y_target_end_confidence_list,
            base_path=base_path,
            idx_to_info=idx_to_info,
            **kwargs,
        )
