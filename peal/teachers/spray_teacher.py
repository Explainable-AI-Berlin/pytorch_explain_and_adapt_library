import json
import multiprocessing
import os
import pathlib
import time
from typing import Union

import h5py
import torch
import numpy as np
import yaml
from corelay.pipeline.spectral import SpectralClustering
from corelay.processor.base import Processor
from corelay.processor.clustering import KMeans, AgglomerativeClustering
from corelay.processor.embedding import EigenDecomposition, TSNEEmbedding, UMAPEmbedding
from corelay.processor.flow import Sequential, Parallel
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from pydantic import BaseModel
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from virelay.model import Workspace
from virelay.server import Server
from zennit.composites import EpsilonPlusFlat
from zennit.torchvision import ResNetCanonizer

from peal.adaptors.clarc import get_layer_name
from peal.architectures.interfaces import TaskConfig
from peal.data.dataset_factory import get_datasets
from peal.data.interfaces import DataConfig
from peal.teachers.interfaces import TeacherInterface


class SprayConfig(BaseModel):
    base_dir: str
    data: DataConfig
    model: str
    port: int = 8080
    use_relative_concept_importance: bool = False
    sum_attribution_channels: bool = False
    normalize_attribtions: bool = True
    attribution_layer: Union[int, list[int]] = -2
    class_idx: Union[int, list[int]] = 0
    num_clusters_max: int = 8
    num_eigval: int = 32
    task: TaskConfig


class SprayTeacher(TeacherInterface):
    def __init__(self, config: Union[dict, SprayConfig], device: str = None):

        if not isinstance(config, SprayConfig):
            config = SprayConfig(**config)
        self.config = config

        self.base_dir = config.base_dir
        pathlib.Path(self.base_dir).mkdir(exist_ok=True)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = torch.load(config.model, map_location=self.device)
        if hasattr(self.model, 'model'):
            self.model = self.model.model
        self.model.eval()

        self.attribution_layer = self.config.attribution_layer
        if isinstance(self.attribution_layer, int):
            self.attribution_layer = [self.attribution_layer]

        self.class_idx = self.config.class_idx
        if isinstance(self.class_idx, int):
            self.class_idx = [self.class_idx]

        self.dataset_class = config.data.dataset_class

        self.analysis_dir = None
        self.attribution_db_path = None
        self.heatmaps_db_path = None
        self.concept_importance_db_path = None
        self.input_db_path = None
        self.analysis_db_path = None
        self.label_map_file_path = None
        self.virelay_project_path = None

    def get_feedback(self, **args) -> dict[tuple[int, int], dict[str, list[str]]]:
        train_data, val_data, test_data = get_datasets(self.config.data, task_config=self.config.task, return_dict=True)
        train_data.url_enabled = True
        val_data.url_enabled = True
        test_data.url_enabled = True

        group_labels = {}
        for class_idx in self.class_idx:

            train_data.disable_class_restriction()
            val_data.disable_class_restriction()
            test_data.disable_class_restriction()
            train_data.enable_class_restriction(class_idx)
            val_data.enable_class_restriction(class_idx)
            test_data.enable_class_restriction(class_idx)
            data = DataLoader(ConcatDataset([train_data, val_data, test_data]), batch_size=64, shuffle=False)

            for layer_idx in self.attribution_layer:
                group_label_map = self._get_feedback(data, layer_idx, class_idx)
                group_labels[(class_idx, layer_idx)] = group_label_map

        return group_labels

    def _get_feedback(self, dataloader: DataLoader, layer: int, class_idx: int) -> dict[str, list[str]]:
        layer_name = get_layer_name(self.model, layer)
        self.analysis_dir = os.path.join(self.base_dir, f"attrbs-layer-{layer}_class-{class_idx}_analysis")
        pathlib.Path(self.analysis_dir).mkdir(exist_ok=True)

        self._compute_attributions(dataloader, layer_name, class_idx)
        self.analysis_db_path = self._spectral_clustering(class_idx)
        self.label_map_file_path = self._create_label_map(class_idx)
        self.virelay_project_path = self._create_project(self.config.data.input_size)
        confounder_ids, non_confounder_ids = self._run_virelay()

        with h5py.File(self.input_db_path, 'r') as input_db:
            filenames = input_db['filenames'].asstr()[:]

        group_label_map = {
            "confounders": [filenames[id] for id in confounder_ids],
            "non_confounders": [filenames[id] for id in non_confounder_ids]
        }
        print(f"found {len(group_label_map['confounders'])} confouders and {len(group_label_map['non_confounders'])} non-confounders")
        group_label_file = os.path.join(self.base_dir, f"group_labels_attrbs-layer-{layer}_class-{class_idx}.json")
        with open(group_label_file, 'w', encoding='utf-8') as file:
            json.dump(group_label_map, file)
        return group_label_map


    def _compute_attributions(self, dataloader: DataLoader, attribution_layer: str, class_idx: int) -> dict[int, str]:

        self.attribution_db_path = os.path.join(self.analysis_dir, "attribution.h5")
        self.heatmaps_db_path = os.path.join(self.analysis_dir, "heatmaps.h5")
        self.concept_importance_db_path = os.path.join(self.analysis_dir, "concept_importance.h5")
        self.input_db_path = os.path.join(self.analysis_dir, "input.h5")

        if pathlib.Path(self.attribution_db_path).is_file():
            return

        attribution_db_file = None
        input_db_file = None
        heatmaps_db_file = None
        concept_importance_db_file = None

        attribution = CondAttribution(self.model)
        canonizer = ResNetCanonizer()
        composite = EpsilonPlusFlat([canonizer])
        cc = ChannelConcept()

        number_samples_processed = 0
        with tqdm(dataloader) as pbar:
            pbar.set_description(f"computing relevances at layer {attribution_layer}...")
            for batch in pbar:
                x = batch["x"].to(self.device).requires_grad_()
                y = batch["y"]
                if len(y.shape) == 2:
                    y = y[:, 0]

                condition = [{"y": class_idx}] * len(x)
                attr = attribution(x, condition, composite, record_layer=[attribution_layer], init_rel=1)

                attributions = attr.relevances[attribution_layer or attribution_layer]
                concept_importance = cc.attribute(attributions)[:, None]
                heatmaps = attr.heatmap[:, None]

                if number_samples_processed == 0:
                    number_classes = attr.prediction.shape[1]
                    number_samples = len(dataloader.dataset)
                    attribution_db_file = create_attribution_database(self.attribution_db_path, attributions[0].shape, number_classes, number_samples)
                    heatmaps_db_file = create_attribution_database(self.heatmaps_db_path, heatmaps[0].shape, number_classes, number_samples)
                    concept_importance_db_file = create_attribution_database(self.concept_importance_db_path, concept_importance[0].shape, number_classes, number_samples)
                    input_db_file = create_input_database(self.input_db_path, x[0].shape, len(dataloader.dataset))

                append_attributions(attribution_db_file, number_samples_processed, attributions, attr.prediction, y)
                append_attributions(heatmaps_db_file, number_samples_processed, heatmaps, attr.prediction, y)
                append_attributions(concept_importance_db_file, number_samples_processed, concept_importance, attr.prediction, y)
                append_inputs(input_db_file, number_samples_processed, x.detach().cpu(), y, batch["url"])
                number_samples_processed += len(y)

        attribution_db_file.close()
        heatmaps_db_file.close()
        concept_importance_db_file.close()
        input_db_file.close()

    def _spectral_clustering(self, class_index: int):

        analysis_db_path = os.path.join(self.analysis_dir, f"analysis.h5")
        if pathlib.Path(analysis_db_path).is_file():
            return analysis_db_path

        number_of_clusters_list = list(range(2, self.config.num_clusters_max + 1))
        pipeline = SpectralClustering(
            preprocessing=Sequential(self._get_preprocessing_pipeline()),
            embedding=EigenDecomposition(n_eigval=self.config.num_eigval, is_output=True),
            clustering=Parallel([
                Parallel([
                    KMeans(n_clusters=number_of_clusters, kwargs={"max_iter": 500}) for number_of_clusters in number_of_clusters_list
                ], broadcast=True),
                Parallel([
                    AgglomerativeClustering(n_clusters=number_of_clusters) for number_of_clusters in number_of_clusters_list
                ], broadcast=True),
                TSNEEmbedding(),
                UMAPEmbedding()
            ], broadcast=True, is_output=True)
        )

        class_name = CLASS_NAMES[self.dataset_class][class_index]

        attribution_path = self.concept_importance_db_path if self.config.use_relative_concept_importance else self.attribution_db_path
        with h5py.File(attribution_path, 'r') as attributions_file:
            labels = attributions_file['label'][:]
            indices_of_samples_in_class, = np.nonzero(labels == class_index)
            attribution_data = attributions_file['attribution'][indices_of_samples_in_class, :]

        (eigenvalues, embedding), (kmeans, ac, tsne, umap) = pipeline(attribution_data)

        with h5py.File(analysis_db_path, 'w') as analysis_file:

            analysis_name = class_name

            analysis_group = analysis_file.require_group(analysis_name)
            analysis_group['index'] = indices_of_samples_in_class.astype('uint32')

            embedding_group = analysis_group.require_group('embedding')
            embedding_group['spectral'] = embedding.astype(np.float32)
            embedding_group['spectral'].attrs['eigenvalue'] = eigenvalues.astype(np.float32)

            embedding_group['tsne'] = tsne.astype(np.float32)
            embedding_group['tsne'].attrs['embedding'] = 'spectral'
            embedding_group['tsne'].attrs['index'] = np.array([0, 1])

            embedding_group['umap'] = umap.astype(np.float32)
            embedding_group['umap'].attrs['embedding'] = 'spectral'
            embedding_group['umap'].attrs['index'] = np.array([0, 1])

            cluster_group = analysis_group.require_group('cluster')
            for number_of_clusters, clustering in zip(number_of_clusters_list, kmeans):
                clustering_dataset_name = f'kmeans-{number_of_clusters:02d}'
                cluster_group[clustering_dataset_name] = clustering
                cluster_group[clustering_dataset_name].attrs['embedding'] = 'spectral'
                cluster_group[clustering_dataset_name].attrs['k'] = number_of_clusters
                cluster_group[clustering_dataset_name].attrs['index'] = np.arange(
                    embedding.shape[1],
                    dtype=np.uint32
                )

            for number_of_clusters, clustering in zip(number_of_clusters_list, ac):
                clustering_dataset_name = f'AgglomerativeClustering-{number_of_clusters:02d}'
                cluster_group[clustering_dataset_name] = clustering
                cluster_group[clustering_dataset_name].attrs['embedding'] = 'spectral'
                cluster_group[clustering_dataset_name].attrs['k'] = number_of_clusters
                cluster_group[clustering_dataset_name].attrs['index'] = np.arange(
                    embedding.shape[1],
                    dtype=np.uint32
                )
        return analysis_db_path

    def _create_label_map(self, class_idx: int) -> str:
        label_map_file_path = os.path.join(self.analysis_dir, "label_map.json")
        if pathlib.Path(label_map_file_path).is_file():
            return label_map_file_path

        class_names = CLASS_NAMES[self.dataset_class]
        label_map = [{
            'index': class_idx,
            'name': class_names[class_idx],
            'word_net_id': str(class_idx)
        }]
        with open(label_map_file_path, 'w', encoding='utf-8') as label_map_file:
            json.dump(label_map, label_map_file)

        return label_map_file_path

    def _create_project(self, input_size: tuple[int]) -> str:

        output_file_path = os.path.join(self.analysis_dir, "virelay_project.yml")
        if pathlib.Path(output_file_path).is_file():
            return output_file_path

        project = {
            'project': {
                'name': "Finding Confounding Factors",
                'model': "PyTorch ResNet-18",
                'label_map': os.path.split(self.label_map_file_path)[1],
                'dataset': {
                    'name': self.dataset_class,
                    'type': 'hdf5',
                    'path': os.path.split(self.input_db_path)[1],
                    'input_width': input_size[1],
                    'input_height': input_size[2],
                    'down_sampling_method': None,
                    'up_sampling_method': None,
                },
                'attributions': {
                    'attribution_method': "Conditional Attribution EpsilonPlusFlat",
                    'attribution_strategy': 'true_label',
                    'sources': [os.path.split(self.heatmaps_db_path)[1]],
                },
                'analyses': [
                    {
                        'analysis_method': "Spectral",
                        'sources': [os.path.split(self.analysis_db_path)[1]],
                    }
                ]
            }
        }

        with open(output_file_path, 'w', encoding='utf-8') as project_file:
            yaml.dump(project, project_file, default_flow_style=False)
        return output_file_path

    def _run_virelay(self) -> tuple[list[int], list[int]]:
        confounder_samples = os.path.join(self.analysis_dir, "confounders.json")
        non_confounder_samples = os.path.join(self.analysis_dir, "non-confounders.json")

        if not (os.path.exists(confounder_samples) and os.path.exists(non_confounder_samples)):

            host_name = "0.0.0.0"
            workspace = Workspace()
            workspace.add_project(self.virelay_project_path)
            app = Server(workspace)
            proc = multiprocessing.Process(
                target=lambda: app.run(host=host_name, port=self.config.port), args=())
            proc.start()
            print('ViReLay GUI is active on localhost:' + str(self.config.port))
            print(f'First, select samples containing the confounding feature and export to {confounder_samples}\n'
                  f'Then, select samples without the confounding feature and export to {non_confounder_samples}')

            while not (os.path.exists(confounder_samples) and os.path.exists(non_confounder_samples)):
                time.sleep(4.0)

            proc.terminate()

        confounder_samples = json.load(open(confounder_samples, 'r'))
        non_confounder_samples = json.load(open(non_confounder_samples, 'r'))
        return confounder_samples['selectedDataPointIndices'], non_confounder_samples['selectedDataPointIndices']

    def _get_preprocessing_pipeline(self) -> list[Processor]:
        preprocessing = []
        if self.config.sum_attribution_channels:
            preprocessing.append(SumChannel())
        if self.config.normalize_attribtions:
            preprocessing.append(Normalize())
        preprocessing.append(Flatten())
        return preprocessing

def create_input_database(dataset_file_path, samples_shape, number_of_samples):

    dataset_file = h5py.File(dataset_file_path, 'w')
    dataset_file.create_dataset(
        'data',
        shape=(number_of_samples,) + tuple(samples_shape),
        dtype='float32'
    )
    dataset_file.create_dataset(
        'label',
        shape=(number_of_samples,),
        dtype='uint16'
    )
    dataset_file.create_dataset(
        'filenames',
        shape=(number_of_samples,),
        dtype=h5py.string_dtype()
    )
    return dataset_file

def append_inputs(dataset_file, index, sample, label, filenames):
    dataset_file['data'][index:sample.shape[0] + index] = sample
    dataset_file['label'][index:label.shape[0] + index] = label
    dataset_file['filenames'][index:len(filenames) + index] = filenames

def create_attribution_database(attribution_database_file_path,
                                attribution_shape,
                                number_of_classes,
                                number_of_samples):

    attribution_database_file = h5py.File(attribution_database_file_path, 'w')
    attribution_database_file.create_dataset(
        'attribution',
        shape=(number_of_samples,) + tuple(attribution_shape),
        dtype='float32'
    )
    attribution_database_file.create_dataset(
        'prediction',
        shape=(number_of_samples, number_of_classes),
        dtype='float32'
    )
    attribution_database_file.create_dataset(
        'label',
        shape=(number_of_samples,),
        dtype='uint16'
    )
    return attribution_database_file

def append_attributions(attribution_database_file,
                        index,
                        attributions,
                        predictions,
                        labels):

    attribution_database_file['attribution'][index:attributions.shape[0] + index] = attributions.detach().numpy()
    attribution_database_file['prediction'][index:predictions.shape[0] + index] = predictions.detach().numpy()
    attribution_database_file['label'][index:labels.shape[0] + index] = labels.detach().numpy()

CLASS_NAMES = {
    "SquareDataset": ['Low Foreground Intensity','High Foreground Intensity'],
    "MnistDataset": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
}

class Flatten(Processor):
    def function(self, data):
        return data.reshape(data.shape[0], np.prod(data.shape[1:]))


class Normalize(Processor):
    def function(self, data):
        data = data / data.sum(tuple(range(1, len(data.shape))), keepdims=True)
        return data


class SumChannel(Processor):
    def function(self, data):
        return data.sum(1)
