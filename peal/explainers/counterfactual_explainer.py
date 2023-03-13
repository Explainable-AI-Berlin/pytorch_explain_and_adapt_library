import torch
import torchvision

from torch import nn
from tqdm import tqdm

from peal.architectures.generators import Glow
from peal.architectures.interfaces import InvertibleGenerator
from peal.utils import (
    load_yaml_config
)
from peal.explainers.lrp_explainer import LRPExplainer

class CounterfactualExplainer:
    def __init__(
        self,
        downstream_model,
        generator,
        explainer_config = '$PEAL/configs/explainers/counterfactual_default.yaml',
        target_confidence_goal = None,
        num_classes = 2
    ):
        self.downstream_model = downstream_model
        self.generator = generator
        self.explainer_config = load_yaml_config(explainer_config)
        self.device = 'cuda' if next(self.downstream_model.parameters()).is_cuda else 'cpu'
        self.loss = torch.nn.CrossEntropyLoss()
        self.lrp_explainer = LRPExplainer(self.downstream_model, num_classes)

    def explain_batch(self, img_batch_in, target_classes, target_confidence_goal_in = None, source_classes = None):
        '''

        '''
        if target_confidence_goal_in is None:
            target_confidence_goal = self.explainer_config['target_confidence_goal']

        else:
            target_confidence_goal = target_confidence_goal_in

        if self.explainer_config['img_regularization'] > 0:
            lrp_heatmaps = self.lrp_explainer.explain_batch(img_batch_in, source_classes)[0]
            regularization_mask = torch.pow(lrp_heatmaps + 0.0000000001, -1) #torch.ones_like(lrp_heatmaps) - lrp_heatmaps
            regularization_mask = regularization_mask.to(self.device)

        img_batch = img_batch_in.clone()
        if isinstance(self.generator, InvertibleGenerator):
            img_batch = img_batch - 0.5

        v_original = self.generator.encode(img_batch.to(self.device))
        if isinstance(v_original, list):
            v = []
            
            for v_org in v_original:
                v.append(nn.Parameter(torch.clone(v_org.detach().cpu()), requires_grad = True))

        else:
            v = nn.Parameter(torch.clone(v_original.detach().cpu()), requires_grad = True)
            v = [v]

        if self.explainer_config['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(v, lr=self.explainer_config['learning_rate'])

        elif self.explainer_config['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(v, lr=self.explainer_config['learning_rate'])

        target_confidences = [0.0 for i in range(len(target_classes))]

        with tqdm(range(self.explainer_config['gradient_steps'])) as pbar:
            for i in pbar:
                if self.explainer_config['use_masking']:
                    mask = torch.tensor(target_confidences).to(self.device) < target_confidence_goal
                    if torch.sum(mask) == 0.0:
                        break
                    
                latent_code = [v_elem.to(self.device) for v_elem in v]
                    
                optimizer.zero_grad()

                img = self.generator.decode(latent_code)

                if isinstance(self.generator, InvertibleGenerator):
                    img = torch.clamp(img + 0.5, 0.0, 1.0)

                logits = self.downstream_model(img + self.explainer_config['img_noise_injection'] * torch.randn_like(img))
                loss = self.loss(logits, target_classes.to(self.device))
                l1_losses = []
                for v_idx in range(len(v_original)):
                    l1_losses.append(torch.mean(torch.abs(v[v_idx].to(self.device) - torch.clone(v_original[v_idx]).detach())))

                loss += self.explainer_config['l1_regularization'] * torch.mean(torch.stack(l1_losses))
                loss += self.explainer_config['log_prob_regularization'] * torch.mean(self.generator.log_prob_z(latent_code))

                if self.explainer_config['img_regularization'] > 0:
                    difference = img - torch.clone(img_batch).to(self.device).detach()
                    loss += torch.mean(torch.abs(difference) * regularization_mask)

                logit_confidences = torch.nn.Softmax()(logits).detach().cpu()
                target_confidences = [float(logit_confidences[i][target_classes[i]]) for i in range(len(target_classes))]

                pbar.set_description(
                    'Creating Counterfactuals: it: ' + str(i) \
                    + ', loss: ' + str(loss.detach().item()) \
                    + ', target_confidences: ' + str(list(target_confidences)[:2]) \
                    + ', visual_difference: ' + str(torch.mean(torch.abs(img_batch_in - img.detach().cpu())).item())
                )

                loss.backward()

                if self.explainer_config['use_masking']:
                    for sample_idx in range(len(target_confidences)):
                        if target_confidences[sample_idx] >= target_confidence_goal:
                            for variable_idx, v_elem in enumerate(v):
                                if self.explainer_config['optimizer'] == 'Adam':
                                    optimizer = torch.optim.Adam(v, lr=self.explainer_config['learning_rate'])

                                v_elem.grad[sample_idx].data.zero_()

                optimizer.step()

        latent_code = [v_elem.to(self.device) for v_elem in v]
        counterfactual =  self.generator.decode(latent_code).detach().cpu()
        if isinstance(self.generator, InvertibleGenerator):
            counterfactual = torch.clamp(counterfactual + 0.5, 0.0, 1.0)

        heatmap_red = torch.maximum(torch.tensor(0.0), torch.sum(img_batch_in, dim = 1) - torch.sum(counterfactual, dim = 1))
        heatmap_blue = torch.maximum(torch.tensor(0.0), torch.sum(counterfactual, dim = 1) - torch.sum(img_batch_in, dim = 1))
        if counterfactual.shape[1] == 3:
            heatmap_green = torch.abs(counterfactual[:,0] - img_batch_in[:,0])
            heatmap_green = heatmap_green + torch.abs(counterfactual[:,1] - img_batch_in[:,1])
            heatmap_green = heatmap_green + torch.abs(counterfactual[:,2] - img_batch_in[:,2])
            heatmap_green = heatmap_green - heatmap_red - heatmap_blue
            counterfactual_rgb = counterfactual

        else:
            heatmap_green = torch.zeros_like(heatmap_red)
            img_batch_in = torch.tile(img_batch_in, [1, 3, 1, 1])
            counterfactual_rgb = torch.tile(torch.clone(counterfactual), [1, 3, 1, 1])

        heatmap = torch.stack([heatmap_red, heatmap_green, heatmap_blue], dim = 1)
        if torch.abs(heatmap.sum() - torch.abs(img_batch_in - counterfactual).sum()) > 0.1:
            print('Error: Heatmap does not match counterfactual')

        heatmap_high_contrast = torch.clamp(heatmap / heatmap.max(), 0.0, 1.0)
        result_img_collage = torch.cat([img_batch_in, counterfactual_rgb, heatmap_high_contrast], -1)

        attributions = []
        for v_idx in range(len(v_original)):
            attributions.append(torch.flatten(v_original[v_idx].detach().cpu() - v[v_idx].detach().cpu(), 1))
        
        attributions = torch.cat(attributions, 1) #.unsqueeze(-1).unsqueeze(-1)
        attributions = torch.reshape(attributions, list(img_batch_in.shape))

        return result_img_collage, counterfactual.detach().cpu(), heatmap, target_confidences, attributions