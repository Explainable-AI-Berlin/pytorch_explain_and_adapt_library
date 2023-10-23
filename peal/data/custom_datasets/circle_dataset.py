import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToTensor
from peal.data.datasets import SymbolicDataset
from peal.generators.interfaces import Generator


class CircleDataset(SymbolicDataset):

    def __init__(self, data_dir, mode, config, transform=ToTensor(), task_config=None, **kwargs):
        super(CircleDataset, self).__init__(data_dir=data_dir, mode=mode, config=config, transform=transform,
                                            task_config=task_config, **kwargs)

    @staticmethod
    def circle_fid(samples):
        radius = 1
        return (((samples.pow(2)).sum(dim=-1) - radius).pow(2)).mean()

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

        ecdf = torch.arange(self.config['num_samples']) / self.config['num_samples']
        true_cdf = (sample_thetas[:, None] >= true_thetas[None, :]).sum(-1) / len(true_data)
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

        input_idx = [idx for idx, element in enumerate(self.attributes) if
                     element not in self.config.confounding_factors]

        # plotting counterfactuals
        plt.figure()
        plt.scatter(data[:, input_idx[0]], data[:, input_idx[1]], color='lightgray')

        counterfactual_path = os.path.join(base_path, str(start_idx))
        Path(counterfactual_path).mkdir(parents=True, exist_ok=True)
        for i, point in enumerate(x_counterfactual_list):
            plt.scatter(x_list[i][0], x_list[i][1], color='green')
            plt.scatter(point[0], point[1], color='red', label='end')
            plt.arrow(
                x_list[i][0], x_list[i][1],
                point[0] - x_list[i][0],
                point[1] - x_list[i][1],
                head_width=0.05, head_length=0.05, fc='blue', ec='blue',

            )
        plt.show()
        plt.savefig(counterfactual_path + '/counterfactuals.png')
        collage_paths.append(counterfactual_path)

        # plotting the train dataset
        data_path = os.path.join(base_path, 'data.png')
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
            plt.scatter(data[:, input_idx[0]], data[:, input_idx[1]], color='lightgray')
            # plt.scatter(df[:, 0], df[:, 1], c=np.where(df[:, -1] == 0, 'green', 'red'))
            plt.scatter(xs[:, 0], xs[:, 1], c=np.where(ys == 0, 'green', 'red'))
            plt.show()
            plt.savefig(data_path)
        # except FileNotFoundError:
        #    pass

        # plotting gradient scalar field
        grad_path = os.path.join(base_path, 'gradient_field.png')
        collage_paths.append(grad_path)
        fig, axs = plt.subplots(1, 2, figsize=(20, 7))
        xx1, xx2 = np.meshgrid(*[
            np.linspace(float(data[:, [idx]].min() - 0.5), float(data[:, [idx]].max() + 0.5), 20)
            for idx in input_idx])
        grid = torch.from_numpy(np.array([xx1.flatten(), xx2.flatten()]).T).to(torch.float32)
        grid.requires_grad = True
        for i in range(2):
            input_data = torch.nn.Parameter(grid.clone())
            logits = classifier(input_data)
            logits[:, i].sum().backward()
            input_gradients = input_data.grad
            axs[i].quiver(input_data[:, 0].detach(), input_data[:, 1].detach(), input_gradients[:, 0],
                          input_gradients[:, 1])
            input_data.grad.zero_()
            axs[i].set_title(f'Class:{i}')
        plt.show()
        plt.savefig(grad_path)

        # plotting contours

        contour_path = os.path.join(base_path, 'contours.png')

        xx1, xx2 = np.meshgrid(*[
            np.linspace(float(data.data[:, [input_idx]].min() - 0.5), float(data.data[:, [input_idx]].max() + 0.5), 200)
            for idx in input_idx])

        grid = torch.from_numpy(np.array([xx1.flatten(), xx2.flatten()]).T).to(torch.float32)
        contour_logits = classifier(grid).detach()
        contour_logits_diff = (contour_logits[:, 1] - contour_logits[:, 0]).reshape(xx1.shape)
        plt.figure()
        plt.scatter(data[:, input_idx[0]], data[:, input_idx[1]], color='lightgray')
        plt.contour(xx1, xx2, contour_logits_diff,
                    levels=torch.linspace(contour_logits_diff.min(), contour_logits_diff.max(), 10).tolist(),
                    lcmap='coolwarm')
        plt.contour(xx1, xx2, contour_logits_diff, levels=[0], colors='red', label='level 0')
        plt.text(1.0, 1.0, 'level 0: red line')
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
