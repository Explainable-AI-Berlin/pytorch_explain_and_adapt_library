import os
import torch
import torchvision
import copy
import yaml
import shutil
import numpy as np
import matplotlib
import platform

from torch import nn
from tensorboardX import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image

from peal.architectures.interfaces import InvertibleGenerator
from peal.utils import (
    load_yaml_config,
    embed_numberstring
)
from peal.data.dataloaders import (
    DataStack,
    DataloaderMixer,
    create_dataloaders_from_datasource
)
from peal.architectures.generators import Glow
from peal.training.trainers import ModelTrainer, calculate_test_accuracy
from peal.explainers.counterfactual_explainer import CounterfactualExplainer
from peal.data.datasets import Image2MixedDataset
from peal.visualization.model_comparison import create_comparison
from peal.teachers.teacher_interface import TeacherInterface
from peal.teachers.human2model_teacher import Human2ModelTeacher
from peal.teachers.model2model_teacher import Model2ModelTeacher
from peal.teachers.segmentation_mask_teacher import SegmentationMaskTeacher
from peal.teachers.virelay_teacher import VirelayTeacher

matplotlib.use('Agg')

class CounterfactualKnowledgeDistillation:
    def __init__(
            self,
            student,
            datasource,
            output_size = None,
            generator = '$PEAL/configs/models/default_generator.yaml',
            base_dir = os.path.join('peal_runs', 'counterfactual_knowledge_distillation'),
            teacher = 'Human@8000',
            adaptor_config = '$PEAL/configs/adaptors/counterfactual_knowledge_distillation_default.yaml',
            gigabyte_vram = None,
            overwrite = False,
            use_visualization = True
    ):
        #
        self.base_dir = base_dir
        #
        self.original_student = student
        self.original_student.eval()
        self.device = 'cuda' if next(self.original_student.parameters()).is_cuda else 'cpu'
        self.overwrite = overwrite
        self.use_visualization = use_visualization
        if self.overwrite or not os.path.exists(os.path.join(self.base_dir, 'config.yaml')):
            self.adaptor_config = load_yaml_config(adaptor_config)
            self.student = copy.deepcopy(student)

        else:
            self.adaptor_config = load_yaml_config(os.path.join(self.base_dir, 'config.yaml'))
            if os.path.exists(os.path.join(self.base_dir, 'model.cpl')):
                self.student = torch.load(os.path.join(self.base_dir, 'model.cpl'), map_location = self.device)

            else:
                self.student = copy.deepcopy(student)

        self.student.eval()

        if not output_size is None:
            self.output_size = output_size
            self.adaptor_config['data']['output_size'] = output_size

        else:
            assert self.adaptor_config['data']['output_size'] != 'None'
        
        def integrate_task_into_adaptor_config(dataset, adaptor_config):
            if hasattr(dataset, 'task_config') and not dataset.task_config is None:
                adaptor_config['task'] = dataset.task_config

            elif isinstance(dataset, Image2MixedDataset):
                adaptor_config['task']['selection'] = [dataset.config['confounding_factors'][0]]
                adaptor_config['task']['output_size'] = 2
                dataset.task_config = adaptor_config['task']
        
        if isinstance(datasource[0], torch.utils.data.Dataset):
            integrate_task_into_adaptor_config(datasource[0], self.adaptor_config)
            X, y = datasource[0].__getitem__(0)
            self.adaptor_config['data']['input_size'] = list(X.shape)
        
        elif isinstance(datasource[0], torch.utils.data.DataLoader):
            integrate_task_into_adaptor_config(datasource[0].dataset, self.adaptor_config)
            X, y = datasource[0].dataset.__getitem__(0)
            self.adaptor_config['data']['input_size'] = list(X.shape)
        
        else:
            assert self.adaptor_config['data']['input_size'] != 'None'
        
        self.input_size = self.adaptor_config['data']['input_size']

        if 'base_batch_size' in self.adaptor_config.keys():
            multiplier = float(np.prod(self.adaptor_config['assumed_input_size']) / np.prod(self.adaptor_config['data']['input_size']))
            if not gigabyte_vram is None and 'gigabyte_vram' in self.adaptor_config.keys():
                multiplier = multiplier * (gigabyte_vram / self.adaptor_config['gigabyte_vram'])
            
            batch_size_adapted = max(1, int(self.adaptor_config['base_batch_size'] * multiplier))
            if self.adaptor_config['batch_size'] == -1:
                self.adaptor_config['batch_size'] = batch_size_adapted
                self.adaptor_config['num_batches'] = int(self.adaptor_config['samples_per_iteration'] / batch_size_adapted) + 1

        #
        self.enable_hints = bool(teacher == 'SegmentationMask')
        self.adaptor_config['data']['has_hints'] = self.enable_hints
        self.train_dataloader, self.val_dataloader, self.test_dataloader = create_dataloaders_from_datasource(
            datasource = datasource,
            config = self.adaptor_config,
            enable_hints = self.enable_hints,
            gigabyte_vram = gigabyte_vram
        )
        self.dataloaders_val = [self.val_dataloader, None]

        # in case the used dataloader has a non-default data normalization it is assumed the inverse function of this normalization is attribute of the underlying dataset
        if hasattr(self.train_dataloader.dataset, 'project_to_pytorch_default'):
            self.project_to_pytorch_default = self.train_dataloader.dataset.project_to_pytorch_default

        else:
            self.project_to_pytorch_default = lambda x: x

        #
        if isinstance(generator, InvertibleGenerator):
            self.generator = generator

        else:
            generator_config = load_yaml_config(generator)
            generator_config['data'] = self.adaptor_config['data']
            self.generator = Glow(generator_config).to(self.device)
            generator_trainer = ModelTrainer(
                config = generator_config, 
                model = self.generator,
                datasource = (self.train_dataloader.dataset, self.dataloaders_val[0].dataset),
                base_dir = base_dir,
                model_name = 'generator',
                gigabyte_vram = gigabyte_vram
            )
            print('Train generator model!')
            generator_trainer.fit()

        self.generator.eval()

        #
        if isinstance(teacher, TeacherInterface):
            self.teacher = teacher

        elif isinstance(teacher, nn.Module):
            teacher.eval()
            self.teacher = Model2ModelTeacher(teacher)

        elif teacher[:5] == 'human':
            if len(teacher) == 10:
                port = int(teacher[-4:])

            else:
                port = 8000

            self.teacher = Human2ModelTeacher(port)
        
        elif teacher == 'SegmentationMask':
            self.teacher = SegmentationMaskTeacher(
                self.adaptor_config['attribution_threshold'])

        elif teacher[:7] == 'virelay':
            self.teacher = VirelayTeacher(
                num_classes = self.output_size,
                port = int(teacher[-4:])
            )

        else:
            self.teacher = Model2ModelTeacher(torch.load(os.path.join(teacher, 'model.cpl')))

        self.dataloader_mixer = DataloaderMixer(self.adaptor_config['training'], self.train_dataloader)
        self.datastack = DataStack(self.train_dataloader, self.output_size)
        self.explainer = CounterfactualExplainer(self.student, self.generator, self.adaptor_config['explainer'])
        self.logits_to_prediction = lambda logits: logits.argmax(-1)

    def generate_counterfactuals_iteration(
            self,
            num_samples,
            error_distribution,
            confidence_score_stats,
            finetune_iteration,
            sample_idx_iteration,
            target_confidence_goal = None
    ):
        collage_paths = []
        counterfactuals = []
        heatmaps = []
        source_classes = []
        target_classes = []
        hints = []
        attributions = []
        ys = []
        num_batches = int(num_samples / self.adaptor_config['batch_size']) + 1

        for batch_idx in range(num_batches):
            print(str(batch_idx) + '/' + str(num_batches))
            current_img_batch = []
            source_classes_current_batch = []
            target_classes_current_batch = []
            ys_current_batch = []
            start_target_confidences = []
            current_hint_batch = []
            sample_idx = 0
            with tqdm(range(100 * self.train_dataloader.dataset.__len__())) as pbar:
                for i in pbar:
                    if sample_idx >= self.adaptor_config['batch_size']:
                        break

                    cm_idx = error_distribution.sample()
                    # TODO verify that this is actually balancing itself!
                    source_class = int(cm_idx / self.output_size)
                    target_class = int(cm_idx % self.output_size)
                    X, y = self.datastack.pop(int(source_class))
                    if isinstance(self.teacher, SegmentationMaskTeacher):
                        y, hint = y

                    logits = self.student(X.to(self.device).unsqueeze(0)).squeeze(0).detach().cpu()
                    start_target_confidence = torch.nn.Softmax()(logits)[target_class]
                    prediction = self.logits_to_prediction(logits)
                    if prediction == y == source_class and start_target_confidence > confidence_score_stats[source_class][target_class]:
                        current_img_batch.append(X)
                        if isinstance(self.teacher, SegmentationMaskTeacher):
                            current_hint_batch.append(hint)
                        
                        else:
                            current_hint_batch.append(torch.zeros_like(X))

                        source_classes_current_batch.append(source_class)
                        target_classes_current_batch.append(torch.tensor(target_class))
                        ys_current_batch.append(y)
                        start_target_confidences.append(start_target_confidence)
                        sample_idx += 1

                    pbar.set_description(
                        'Sample selection phase: num_samples: ' + str(self.adaptor_config['samples_per_iteration'] - num_samples  \
                            + len(collage_paths)) + ', num_candidates: ' + str(sample_idx)
                    )

            current_img_batch = torch.stack(current_img_batch)
            target_classes_current_batch = torch.stack(target_classes_current_batch)

            #
            result_img_collage, counterfactual, heatmaps_current_batch, end_target_confidences, current_attributions = self.explainer.explain_batch(
                img_batch_in = current_img_batch,
                target_classes = target_classes_current_batch,
                source_classes = torch.tensor(source_classes_current_batch)
            )

            #
            for sample_idx in range(result_img_collage.shape[0]):
                if end_target_confidences[sample_idx] >= self.adaptor_config['explainer']['target_confidence_goal']:
                    current_collage = self.project_to_pytorch_default(result_img_collage[sample_idx])
                    current_collage = torchvision.utils.make_grid(current_collage, nrow = self.adaptor_config['batch_size'])
                    plt.gcf()
                    plt.imshow(current_collage.permute(1,2,0))
                    title_string = str(source_classes_current_batch[sample_idx]) + ' -> ' + str(target_classes_current_batch[sample_idx].item())
                    title_string +=  ', Target: ' + str(round(float(start_target_confidences[sample_idx]), 2)) + ' -> ' + str(round(float(end_target_confidences[sample_idx]), 2))
                    plt.title(title_string)
                    collage_path = os.path.join(self.base_dir, str(finetune_iteration), 'collages', embed_numberstring(str(sample_idx_iteration)) + '.png')
                    plt.axis('off')
                    plt.savefig(collage_path)
                    img_np = np.array(Image.open(collage_path))[:,80:-80]
                    img = Image.fromarray(img_np).resize((3 * self.adaptor_config['data']['input_size'][1], 3 * self.adaptor_config['data']['input_size'][1]))
                    img.save(collage_path)
                    counterfactuals.append(self.project_to_pytorch_default(counterfactual[sample_idx]))
                    heatmaps.append(self.project_to_pytorch_default(heatmaps_current_batch[sample_idx]))
                    collage_paths.append(collage_path)
                    source_classes.append(source_classes_current_batch[sample_idx])
                    target_classes.append(target_classes_current_batch[sample_idx].item())
                    ys.append(ys_current_batch[sample_idx])
                    hints.append(current_hint_batch[sample_idx])
                    attributions.append(current_attributions[sample_idx])
                    sample_idx_iteration += 1

        return collage_paths, counterfactuals, heatmaps, source_classes, target_classes, hints, attributions, ys

    def generate_counterfactuals(self, error_distribution, confidence_score_stats, finetune_iteration):
        if isinstance(self.teacher, SegmentationMaskTeacher):
            for dataloader in self.datastack.datasource.dataloaders:
                dataloader.dataset.enable_hints()

            self.datastack.datasource.reset()

        self.datastack.reset()

        shutil.rmtree(os.path.join(self.base_dir, str(finetune_iteration), 'collages'), ignore_errors = True)
        Path(os.path.join(self.base_dir, str(finetune_iteration), 'collages')).mkdir(parents=True, exist_ok=True)

        collage_paths = []
        counterfactuals = []
        heatmaps = []
        source_classes = []
        target_classes = []
        ys = []
        hints = []
        attributions = []
        continue_collecting = True
        current_acceptance_threshold = self.adaptor_config['explainer']['target_confidence_goal']
        while continue_collecting:
            values = self.generate_counterfactuals_iteration(
                num_samples = self.adaptor_config['samples_per_iteration'],
                error_distribution = error_distribution,
                confidence_score_stats = confidence_score_stats,
                finetune_iteration = finetune_iteration,
                sample_idx_iteration = len(collage_paths),
                target_confidence_goal = current_acceptance_threshold
            )
            collage_paths.extend(values[0])
            counterfactuals.extend(values[1])
            heatmaps.extend(values[2])
            source_classes.extend(values[3])
            target_classes.extend(values[4])
            hints.extend(values[5])
            attributions.extend(values[6])
            ys.extend(values[7])
            print(str(len(collage_paths)) + '/' + str(self.adaptor_config['samples_per_iteration']))
            if len(collage_paths) < self.adaptor_config['samples_per_iteration']:
                if current_acceptance_threshold == 0.51 and len(values[0]) == 0:
                    continue_collecting = False

                elif len(values[0]) < self.adaptor_config['samples_per_iteration'] / 2:
                    current_acceptance_threshold = float(np.maximum(0.51, current_acceptance_threshold - 0.1))

            else:
                continue_collecting = False

        if isinstance(self.teacher, SegmentationMaskTeacher):
            for dataloader in self.datastack.datasource.dataloaders:
                dataloader.dataset.disable_hints()
                
            self.datastack.datasource.reset()

        return collage_paths, counterfactuals, heatmaps, source_classes, target_classes, hints, attributions, ys

    def calculate_validation_statistics(self, finetune_iteration):
        #
        path = os.path.join(self.base_dir, str(finetune_iteration))

        #
        ys = []
        y_preds = []
        targets = []
        start_target_confidences = []
        result_img_collages = []
        counterfactuals = []
        heatmaps = []
        end_target_confidences = []
        attributions = []

        #
        confusion_matrix = np.zeros([self.output_size, self.output_size])
        correct = 0
        num_samples = 0
        # TODO here probably uncertainties should be used
        confidence_scores = []
        for i in range(self.output_size):
            confidence_scores.append([])

        with tqdm(enumerate(self.dataloaders_val[0])) as pbar:
            for it, (X, y) in pbar:
                pred_confidences = torch.nn.Softmax()(self.student(X.to(self.device))).detach().cpu()
                y_pred = self.logits_to_prediction(pred_confidences)
                for i in range(y.shape[0]):
                    if y_pred[i] == y[i]:
                        correct += 1
                        confidence_scores[y[i]].append(pred_confidences[i])

                    confusion_matrix[int(y[i])][int(y_pred[i])] += 1
                    num_samples += 1

                pbar.set_description('Calculate Confusion Matrix: it: ' + str(it) + ', current_accuracy: ' + str(correct / num_samples))

                batch_targets = (y_pred + 1) % self.output_size
                results = self.explainer.explain_batch(
                    img_batch_in = X,
                    target_classes = batch_targets,
                    source_classes = y_pred
                )
                ys.append(y)
                y_preds.append(y_pred)
                targets.append(batch_targets)
                batch_target_start_confidences = []
                for sample_idx in range(pred_confidences.shape[0]):
                    batch_target_start_confidences.append(pred_confidences[sample_idx][batch_targets[sample_idx]])

                start_target_confidences.append(torch.stack(batch_target_start_confidences, 0))
                result_img_collages.append(results[0])
                counterfactuals.append(results[1])
                heatmaps.append(results[2])
                end_target_confidences.append(results[3])
                attributions.append(results[4])

                if num_samples >= self.adaptor_config['max_validation_samples']:
                    break

        confidence_score_stats = []
        for i in range(self.output_size):
            if len(confidence_scores[i]) >= 1:
                confidence_score_stats.append(torch.quantile(
                    torch.stack(confidence_scores[i], dim = 1),
                    self.adaptor_config['min_start_target_percentile'],
                    dim = 1
                ))

            else:
                confidence_score_stats.append(torch.zeros([self.output_size]))

        confidence_score_stats = torch.stack(confidence_score_stats)
        accuracy = correct / num_samples

        if self.adaptor_config['use_confusion_matrix'] and not accuracy == 1.0:
            error_matrix = np.copy(confusion_matrix)
            for i in range(error_matrix.shape[0]):
                error_matrix[i][i] = 0.0

            error_matrix = error_matrix.flatten()
            error_matrix = error_matrix / error_matrix.sum()
            error_distribution = torch.distributions.categorical.Categorical(torch.tensor(error_matrix))

        else:
            error_matrix = torch.ones([self.output_size, self.output_size]) - torch.eye(self.output_size)
            error_matrix = error_matrix.flatten() / error_matrix.sum()
            error_distribution = torch.distributions.categorical.Categorical(error_matrix)

        validation_basics = accuracy, error_distribution, confidence_score_stats
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(path, 'validation_basics.npz'), 'wb') as f:
            np.savez(
                f,
                accuracy = accuracy,
                error_matrix = error_matrix,
                confidence_score_stats = confidence_score_stats
            )

        y_list = []
        y_pred_list = []
        counterfactual_list = []
        heatmap_list = []
        target_class_list = []
        hint_list = []
        attribution_list = []
        collage_path_list = []
        sample_idx_iteration = 0
        Path(os.path.join(self.base_dir, str(finetune_iteration), 'validation_collages')).mkdir(parents=True, exist_ok=True)
        for batch_idx in range(len(ys)):
            for sample_idx in range(counterfactuals[batch_idx].shape[0]):
                if end_target_confidences[batch_idx][sample_idx] > 0.5:
                    y_list.append(ys[batch_idx][sample_idx])
                    y_pred_list.append(y_preds[batch_idx][sample_idx])
                    counterfactual_list.append(counterfactuals[batch_idx][sample_idx])
                    heatmap_list.append(heatmaps[batch_idx][sample_idx])
                    target_class_list.append(targets[batch_idx][sample_idx])
                    # TODO introduce hints here!
                    #hint_list.append(hints[batch_idx][sample_idx])
                    hint_list.append(torch.zeros_like(heatmaps[batch_idx][sample_idx]))
                    attribution_list.append(attributions[batch_idx][sample_idx])
                    current_collage = self.project_to_pytorch_default(result_img_collages[batch_idx][sample_idx])
                    current_collage = torchvision.utils.make_grid(current_collage, nrow = self.adaptor_config['batch_size'])
                    plt.gcf()
                    plt.imshow(current_collage.permute(1,2,0))
                    title_string = str(int(ys[batch_idx][sample_idx])) + ' -> ' + str(targets[batch_idx][sample_idx].item())
                    title_string +=  ', Target: ' + str(round(float(start_target_confidences[batch_idx][sample_idx]), 2)) + ' -> '
                    title_string += str(round(float(end_target_confidences[batch_idx][sample_idx]), 2))
                    plt.title(title_string)
                    collage_path = os.path.join(self.base_dir, str(finetune_iteration), 'validation_collages', embed_numberstring(str(sample_idx_iteration)) + '.png')
                    plt.axis('off')
                    plt.savefig(collage_path)                    
                    img_np = np.array(Image.open(collage_path))[:,80:-80]
                    img = Image.fromarray(img_np).resize((3 * self.adaptor_config['data']['input_size'][1], 3 * self.adaptor_config['data']['input_size'][1]))
                    img.save(collage_path)
                    collage_path_list.append(collage_path)
                    sample_idx_iteration += 1

        with open(os.path.join(path, 'validation_counterfactuals.npz'), 'wb') as f:
            np.savez(
                f,
                y_list = torch.stack(y_list).numpy(),
                y_pred_list = torch.stack(y_pred_list).numpy(),
                counterfactual_list = torch.stack(counterfactual_list).numpy(),
                heatmap_list = torch.stack(heatmap_list).numpy(),
                target_class_list = torch.stack(target_class_list).numpy(),
                hint_list = torch.stack(hint_list).numpy(),
                attribution_list = torch.stack(attribution_list).numpy()
            )

        validation_lists = y_list, y_pred_list, counterfactual_list, heatmap_list, target_class_list, hint_list, attribution_list, collage_path_list

        return validation_basics, validation_lists

    def retrieve_validation_statistics(self, finetune_iteration):
        #
        if not os.path.exists(os.path.join(self.base_dir, str(finetune_iteration), 'validation_counterfactuals.npz')):
            validation_basics, validation_lists = self.calculate_validation_statistics(finetune_iteration)
            accuracy, error_distribution, confidence_score_stats = validation_basics
            y_list, y_pred_list, counterfactual_list, heatmap_list, target_class_list, hint_list, attribution_list, collage_path_list = validation_lists

        else:
            #
            with open(os.path.join(self.base_dir, str(finetune_iteration), 'validation_basics.npz'), 'rb') as f:
                validation_basics = np.load(f, allow_pickle=True)
                accuracy = float(validation_basics['accuracy'])
                error_matrix = torch.tensor(validation_basics['error_matrix'])
                error_distribution = torch.distributions.categorical.Categorical(error_matrix)
                confidence_score_stats = torch.tensor(validation_basics['confidence_score_stats'])
            #
            with open(os.path.join(self.base_dir, str(finetune_iteration), 'validation_counterfactuals.npz'), 'rb') as f:
                validation_counterfactuals = np.load(f, allow_pickle=True)
                y_list = list(torch.tensor(validation_counterfactuals['y_list']))
                y_pred_list = list(torch.tensor(validation_counterfactuals['y_pred_list']))
                counterfactual_list = list(torch.tensor(validation_counterfactuals['counterfactual_list']))
                heatmap_list = list(torch.tensor(validation_counterfactuals['heatmap_list']))
                target_class_list = list(torch.tensor(validation_counterfactuals['target_class_list']))
                hint_list = list(torch.tensor(validation_counterfactuals['hint_list']))
                attribution_list = list(torch.tensor(validation_counterfactuals['attribution_list']))
                collage_path_list = os.listdir(os.path.join(self.base_dir, str(finetune_iteration), 'validation_collages'))
                collage_path_list = list(map(lambda x: os.path.join(self.base_dir, str(finetune_iteration), 'validation_collages', x), collage_path_list))

        if len(counterfactual_list) == 0:
            return accuracy, error_distribution, confidence_score_stats, 0.0, 0.0, 0.0, 0.0, 0.0

        if not os.path.exists(os.path.join(self.base_dir, str(finetune_iteration), 'feedback_validation.txt')):
            feedback = self.teacher.get_feedback(
                counterfactuals = counterfactual_list,
                heatmaps = heatmap_list,
                collage_paths = collage_path_list,
                gt_classes = y_list,
                source_classes = y_pred_list,
                target_classes=target_class_list,
                hints=hint_list,
                attributions=attribution_list,
                base_dir=os.path.join(self.base_dir, str(finetune_iteration), 'validation_teacher'),
            )

            create_dataset(
                counterfactuals = counterfactual_list,
                feedback = feedback,
                source_classes = list(map(lambda x: int(x), y_list)),
                target_classes = list(map(lambda x: int(x), target_class_list)),
                hints = hint_list,
                base_dir = self.base_dir,
                finetune_iteration = finetune_iteration + 1,
                output_size = self.output_size,
                teacher = self.teacher,
                mode = 'validation_'
            )            

            with open(os.path.join(self.base_dir, str(finetune_iteration), 'feedback_validation.txt'), 'w') as f: f.write('\n'.join(feedback))

        else:
            with open(os.path.join(self.base_dir, str(finetune_iteration), 'feedback_validation.txt'), 'r') as f: feedback = f.read().split('\n')

        num_samples = min(self.adaptor_config['max_validation_samples'], len(self.val_dataloader.dataset))
        counterfactual_rate = len(counterfactual_list) / num_samples
        ood_rate = len(list(filter(lambda sample: sample == 'ood', feedback))) / len(feedback)

        num_true_2sided = len(list(filter(lambda sample: sample == 'true', feedback)))
        num_false_2sided = len(list(filter(lambda sample: sample == 'false', feedback)))
        fa_2sided = num_true_2sided / (num_true_2sided + num_false_2sided)

        num_true_1sided = len(list(filter(lambda x: x[1] == 'true' and y_list[x[0]] == y_pred_list[x[0]], enumerate(feedback))))
        num_false_1sided = len(list(filter(lambda x: x[1] == 'false' and y_list[x[0]] == y_pred_list[x[0]], enumerate(feedback))))
        fa_1sided = num_true_1sided / (num_true_1sided + num_false_1sided)
        fa_absolute = num_true_1sided / num_samples

        return accuracy, error_distribution, confidence_score_stats, counterfactual_rate, ood_rate, fa_2sided, fa_1sided, fa_absolute

    def visualize_progress(self, paths):
        task_config_buffer = copy.deepcopy(self.test_dataloader.dataset.task_config)
        criterions = {}
        if isinstance(self.test_dataloader.dataset, Image2MixedDataset) and 'Confounder' in self.test_dataloader.dataset.attributes:
            self.test_dataloader.dataset.task_config = {'selection' : [], 'criterions' : []}
            criterions['class'] = lambda X, y: int(y[self.test_dataloader.dataset.attributes.index(task_config_buffer['selection'][0])])
            criterions['confounder'] = lambda X, y: int(y[self.test_dataloader.dataset.attributes.index('Confounder')])
            criterions['uncorrected'] = lambda X, y: int(
                self.original_student(X.unsqueeze(0).to(self.device)).squeeze(0).cpu().argmax()
            )
            criterions['cfkd'] = lambda X, y: int(
                self.student(X.unsqueeze(0).to(self.device)).squeeze(0).cpu().argmax()
            )

        else:
            criterions['class'] = lambda X, y: int(y)
            criterions['uncorrected'] = lambda X, y: int(
                self.original_student(X.unsqueeze(0).to(self.device)).squeeze(0).cpu().argmax()
            )
            criterions['cfkd'] = lambda X, y: int(
                self.student(X.unsqueeze(0).to(self.device)).squeeze(0).cpu().argmax()
            )

        img = create_comparison(
            dataset = self.test_dataloader.dataset,
            criterions = criterions,
            columns = {            
                'Counterfactual\nExplanation' : ['cf', self.original_student, 'uncorrected'],
                'CFKD\ncorrected' : ['cf', self.student, 'cfkd'],
                'LRP\nuncorrected' : ['lrp', self.original_student, 'uncorrected'],
                'LRP\ncorrected' : ['lrp', self.student, 'cfkd'],
            },
            score_reference_idx = 1,
            generator = self.generator,
            device = self.device,
            explainer_config = self.adaptor_config['explainer']
        )
        self.test_dataloader.dataset.task_config = task_config_buffer
        for path in paths:
            img.save(path)

        return img

    def run(self):
        '''
        Run the counterfactual knowledge distillation
        '''
        if self.overwrite or not os.path.exists(os.path.join(self.base_dir, '0', 'validation_counterfactuals.npz')):
            assert self.adaptor_config['current_iteration'] == 0
            print('Create base_dir in: ' + str(self.base_dir))
            shutil.rmtree(self.base_dir, ignore_errors = True)
            Path(self.base_dir).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(self.base_dir, 'config.yaml'), 'w') as file: yaml.dump(self.adaptor_config, file)
            with open(os.path.join(self.base_dir, 'platform.txt'), 'w') as f: f.write(platform.node())
            writer = SummaryWriter(os.path.join(self.base_dir, 'logs'))

            if self.output_size == 2 and self.use_visualization: self.visualize_progress([os.path.join(self.base_dir, 'visualization.png')])

            test_accuracy = calculate_test_accuracy(self.student, self.test_dataloader, self.device)
            writer.add_scalar('test_accuracy', test_accuracy, 0)

            val_accuracy, error_distribution, confidence_score_stats, counterfactual_rate, ood_rate, fa_2sided, fa_1sided, fa_absolute = self.retrieve_validation_statistics(0)

        else:
            with open(os.path.join(self.base_dir, 'platform.txt'), 'w') as f: f.write(platform.node())
            writer = SummaryWriter(os.path.join(self.base_dir, 'logs'))
            val_accuracy, error_distribution, confidence_score_stats, counterfactual_rate, ood_rate, fa_2sided, fa_1sided, fa_absolute = self.retrieve_validation_statistics(
                self.adaptor_config['current_iteration']
            )

        writer.add_scalar('val_accuracy', val_accuracy, self.adaptor_config['current_iteration'])
        writer.add_scalar('counterfactual_rate', counterfactual_rate, self.adaptor_config['current_iteration'])
        writer.add_scalar('ood_rate', ood_rate, self.adaptor_config['current_iteration'])
        writer.add_scalar('fa_2sided', fa_2sided, self.adaptor_config['current_iteration'])
        writer.add_scalar('fa_1sided', fa_1sided, self.adaptor_config['current_iteration'])
        writer.add_scalar('fa_absolute', fa_absolute, self.adaptor_config['current_iteration'])

        # iterate over the finetune iterations
        for finetune_iteration in range(self.adaptor_config['current_iteration'] + 1, self.adaptor_config['finetune_iterations'] + 1):
            current_dataset_path = os.path.join(self.base_dir, str(finetune_iteration), 'dataset')

            if not os.path.exists(os.path.join(self.base_dir, str(finetune_iteration), 'counterfactuals.npz')):
                collage_paths, counterfactuals, heatmaps, source_classes, target_classes, hints, attributions, ys = self.generate_counterfactuals(
                    error_distribution = error_distribution,
                    confidence_score_stats = confidence_score_stats,
                    finetune_iteration = finetune_iteration
                )

                if len(collage_paths) < self.adaptor_config['samples_per_iteration']:
                    print('No counterfactuals could be found anymore!')
                    open(os.path.join(self.base_dir, 'warning.txt'), 'w').write('No counterfactuals could be found anymore in iteration ' + str(finetune_iteration) + '!')
                    return self.student

                with open(os.path.join(self.base_dir, str(finetune_iteration), 'counterfactuals.npz'), 'wb') as f:
                    np.savez(
                        f,
                        counterfactuals = torch.stack(counterfactuals).numpy(),
                        heatmaps = torch.stack(heatmaps).numpy(),
                        source_classes = torch.tensor(source_classes).numpy(),
                        target_classes = torch.tensor(target_classes).numpy(),
                        hints = torch.stack(hints).numpy(),
                        attributions = torch.stack(attributions).numpy(),
                        ys = torch.stack(ys).numpy()
                    )

            else:                
                with open(os.path.join(self.base_dir, str(finetune_iteration), 'counterfactuals.npz'), 'rb') as f:
                    validation_counterfactuals = np.load(f, allow_pickle=True)
                    counterfactuals = list(torch.tensor(validation_counterfactuals['counterfactuals']))
                    heatmaps = list(torch.tensor(validation_counterfactuals['heatmaps']))
                    source_classes = list(torch.tensor(validation_counterfactuals['source_classes']))
                    target_classes = list(torch.tensor(validation_counterfactuals['target_classes']))
                    hints = list(torch.tensor(validation_counterfactuals['hints']))
                    attributions = list(torch.tensor(validation_counterfactuals['attributions']))
                    ys = list(torch.tensor(validation_counterfactuals['ys']))
                    collage_paths = os.listdir(os.path.join(self.base_dir, str(finetune_iteration), 'collages'))
                    collage_paths = list(map(lambda x: os.path.join(self.base_dir, str(finetune_iteration), 'collages', x), collage_paths))

            if not os.path.exists(os.path.join(self.base_dir, str(finetune_iteration), 'feedback.txt')):
                feedback = self.teacher.get_feedback(
                    counterfactuals = counterfactuals,
                    heatmaps = heatmaps,
                    collage_paths = collage_paths,
                    gt_classes = ys,
                    source_classes = source_classes,
                    target_classes = target_classes,
                    hints=hints,
                    attributions=heatmaps, #attributions,
                    base_dir = os.path.join(self.base_dir, str(finetune_iteration), 'teacher'),
                )

                create_dataset(
                    counterfactuals = counterfactuals,
                    feedback = feedback,
                    source_classes = source_classes,
                    target_classes = target_classes,
                    hints = hints,                    
                    base_dir = self.base_dir,
                    finetune_iteration = finetune_iteration,
                    output_size = self.output_size,
                    teacher = self.teacher
                )
                with open(os.path.join(self.base_dir, str(finetune_iteration), 'feedback.txt'), 'w') as f: f.write('\n'.join(feedback))

            else:
                with open(os.path.join(self.base_dir, str(finetune_iteration), 'feedback.txt'), 'r') as f: feedback = f.read().split('\n')

            if not os.path.exists(os.path.join(self.base_dir, str(finetune_iteration), 'finetuned_model', 'model.cpl')):
                shutil.rmtree(os.path.join(self.base_dir, str(finetune_iteration), 'finetuned_model'), ignore_errors = True)
                Path(os.path.join(self.base_dir, str(finetune_iteration), 'finetuned_model')).mkdir(parents=True, exist_ok=True)
                #
                data_config = copy.deepcopy(self.adaptor_config)
                data_config['training']['split'] = [1.0, 1.0]
                current_dataloader, _, _ = create_dataloaders_from_datasource(current_dataset_path, data_config)

                #
                val_dataset_path = os.path.join(self.base_dir, str(finetune_iteration), 'validation_dataset')
                validation_data_config = copy.deepcopy(self.adaptor_config)
                validation_data_config['training']['split'] = [0.0, 1.0]
                _, current_dataloader_val, _ = create_dataloaders_from_datasource(current_dataset_path, data_config)

                #
                priority = (1 / (1 - self.adaptor_config['mixing_ratio'])) * self.adaptor_config['mixing_ratio'] * len(self.dataloader_mixer) / len(current_dataloader.dataset)
                self.dataloader_mixer.append(current_dataloader, priority = priority)
                assert abs(self.dataloader_mixer.priorities[-1] - self.adaptor_config['mixing_ratio']) < 0.01, 'priorities do not match! ' + str(self.dataloader_mixer.priorities)
                self.dataloaders_val[1] = current_dataloader_val

                if not self.adaptor_config['continuos_learning']:
                    def weight_reset(m):
                        reset_parameters = getattr(m, "reset_parameters", None)
                        if callable(reset_parameters):
                            m.reset_parameters()

                    self.student.apply(weight_reset)
                    
                finetune_trainer = ModelTrainer(
                    config = copy.deepcopy(self.adaptor_config),
                    model = self.student,
                    datasource = (self.dataloader_mixer, self.dataloaders_val),
                    model_name = 'finetuned_model',
                    base_dir = os.path.join(self.base_dir, str(finetune_iteration)),
                    val_dataloader_weights = [1 - self.adaptor_config['mixing_ratio'], self.adaptor_config['mixing_ratio']]
                )
                finetune_trainer.fit(continue_training = True)

            else:
                self.student = torch.load(os.path.join(self.base_dir, str(finetune_iteration), 'finetuned_model', 'model.cpl'), map_location = self.device)
            
            test_accuracy = calculate_test_accuracy(self.student, self.test_dataloader, self.device)
            writer.add_scalar('test_accuracy', test_accuracy, finetune_iteration)

            val_accuracy, error_distribution, confidence_score_stats, counterfactual_rate, ood_rate, fa_2sided, fa_1sided, fa_absolute = self.retrieve_validation_statistics(
                finetune_iteration
            )
            writer.add_scalar('val_accuracy', val_accuracy, finetune_iteration)
            writer.add_scalar('counterfactual_rate', counterfactual_rate, finetune_iteration)
            writer.add_scalar('ood_rate', ood_rate, finetune_iteration)
            writer.add_scalar('fa_2sided', fa_2sided, finetune_iteration)
            writer.add_scalar('fa_1sided', fa_1sided, finetune_iteration)
            writer.add_scalar('fa_absolute', fa_absolute, finetune_iteration)

            if self.output_size == 2 and self.use_visualization: self.visualize_progress([
                os.path.join(self.base_dir, str(finetune_iteration), 'visualization.png'),
                os.path.join(self.base_dir, 'visualization.png')
            ])

            if self.adaptor_config['replace_model']:
                torch.save(self.student , os.path.join(self.base_dir, 'model.cpl'))

            if fa_1sided > self.adaptor_config['fa_1sided_prime']:
                self.adaptor_config['fa_1sided_prime'] = fa_1sided
                self.adaptor_config['replace_model'] = True

            else:
                self.adaptor_config['replace_model'] = False

            self.adaptor_config['current_iteration'] = self.adaptor_config['current_iteration'] + 1
            with open(os.path.join(self.base_dir, 'config.yaml'), 'w') as file:
                yaml.dump(self.adaptor_config, file)
    
        return self.student


def create_dataset(counterfactuals, feedback, source_classes, target_classes, base_dir, finetune_iteration, output_size, teacher = None, mode = '', hints = None):
    assert len(counterfactuals) == len(feedback) == len(source_classes) == len(target_classes), 'missmatch in list lengths'

    #
    if isinstance(teacher, SegmentationMaskTeacher):
        img_dir = os.path.join(mode + 'dataset', 'imgs')
        Path(os.path.join(base_dir, str(finetune_iteration), mode + 'dataset', 'masks')).mkdir(parents=True, exist_ok=True)
    
    else:
        img_dir = mode + 'dataset'

    for class_name in range(output_size):
        Path(os.path.join(base_dir, str(finetune_iteration), img_dir, str(class_name))).mkdir(parents=True, exist_ok=True)

    #
    current_sample_idx = 0
    dataset_dir = os.path.join(base_dir, str(finetune_iteration), mode + 'dataset')
    for sample_idx in range(len(feedback)):
        if feedback[sample_idx] == 'ood':
            continue

        elif feedback[sample_idx] == 'true':
            sample_name = 'true_' + str(int(source_classes[sample_idx])) + '_to_' + str(int(target_classes[sample_idx])) + '_' + str(current_sample_idx) + '.png'
            if not isinstance(teacher, SegmentationMaskTeacher):
                torchvision.utils.save_image(
                    counterfactuals[sample_idx],
                    os.path.join(
                        dataset_dir,
                        str(int(target_classes[sample_idx])),
                        sample_name
                    )
                )

            else:
                torchvision.utils.save_image(
                    counterfactuals[sample_idx],
                    os.path.join(dataset_dir, 'imgs', str(int(target_classes[sample_idx])), sample_name)
                )
                torchvision.utils.save_image(
                    hints[sample_idx],
                    os.path.join(dataset_dir, 'masks', sample_name)
                )

            current_sample_idx += 1

        elif feedback[sample_idx] == 'false':
            sample_name = 'false_' + str(int(source_classes[sample_idx])) + '_to_' + str(int(target_classes[sample_idx])) + '_' + str(current_sample_idx) + '.png'
            if not isinstance(teacher, SegmentationMaskTeacher):
                torchvision.utils.save_image(
                    counterfactuals[sample_idx],
                    os.path.join(dataset_dir, str(int(source_classes[sample_idx])), sample_name)
                )

            else:
                torchvision.utils.save_image(
                    counterfactuals[sample_idx],
                    os.path.join(dataset_dir, 'imgs', str(int(source_classes[sample_idx])), sample_name)
                )
                torchvision.utils.save_image(
                    hints[sample_idx],
                    os.path.join(dataset_dir, 'masks', sample_name)
                )

            current_sample_idx += 1

        else:
            print(feedback[sample_idx] + ' is impossible feedback!')

    
