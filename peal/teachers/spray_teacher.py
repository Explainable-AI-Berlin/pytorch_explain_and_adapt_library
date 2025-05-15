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
from corelay.processor.clustering import KMeans, AgglomerativeClustering, HDBSCAN, DBSCAN
from corelay.processor.embedding import EigenDecomposition, TSNEEmbedding, UMAPEmbedding
from corelay.processor.flow import Sequential, Parallel
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from pydantic import BaseModel
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm
from virelay.model import Workspace
from virelay.server import Server
from zennit.composites import EpsilonPlusFlat
from zennit.torchvision import ResNetCanonizer

from peal.adaptors.clarc import get_layer_name
from peal.architectures.interfaces import TaskConfig
from peal.data.dataset_factory import get_datasets
from peal.data.interfaces import DataConfig
from peal.global_utils import save_yaml_config
from peal.teachers.interfaces import TeacherInterface


class SprayConfig(BaseModel):
    config_name: str = "SprayConfig"
    base_dir: str
    data: DataConfig
    dataset_name: str = None
    model: str
    port: int = 8080
    use_relative_concept_importance: bool = False
    sum_attribution_channels: bool = False
    normalize_attribtions: bool = True
    attribution_layer: int = -2
    classes_total: int
    num_clusters_max: int = 8
    num_eigval: int = 32
    tsne_perplexity: float = 64.
    umap_neighbors: int = 32
    task: TaskConfig
    skip_wrongly_classified_samples: bool = False
    max_samples: int = 10000


class Spray(TeacherInterface):
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
        self.dataset_class = config.dataset_name or config.data.dataset_class

        self.analysis_dir = None
        self.attribution_db_path = None
        self.heatmaps_db_path = None
        self.concept_importance_db_path = None
        self.input_db_path = None
        self.analysis_db_path = None
        self.label_map_file_path = None
        self.virelay_project_path = None

        self.train_data, self.val_data, self.test_data = get_datasets(self.config.data, task_config=self.config.task, return_dict=True)
        self.train_data.url_enabled = True
        self.val_data.url_enabled = True
        self.test_data.url_enabled = True
        self.train_data.disable_class_restriction()
        self.val_data.disable_class_restriction()
        self.test_data.disable_class_restriction()

    def get_feedback(self, **args) -> dict[str, int]:

        layer_name = get_layer_name(self.model, self.attribution_layer)
        self.analysis_dir = os.path.join(self.base_dir, f"attrbs-layer-{self.attribution_layer}_analysis")
        pathlib.Path(self.analysis_dir).mkdir(exist_ok=True)
        save_yaml_config(self.config, os.path.join(self.analysis_dir, "config.yaml"))

        self._compute_attributions(layer_name)
        self.analysis_db_path = self._spectral_clustering()
        self.label_map_file_path = self._create_label_map()
        self.virelay_project_path = self._create_project(self.config.data.input_size)
        confounder_ids, non_confounder_ids = self._run_virelay()

        with h5py.File(self.input_db_path, 'r') as input_db:
            filenames = input_db['filenames'].asstr()[:]

        group_label_map = {}
        for id in confounder_ids:
            group_label_map[filenames[id]] = 1
        for id in non_confounder_ids:
            group_label_map[filenames[id]] = 0

        labels = np.array(list(group_label_map.values()))
        print(f"found {len(labels[labels == 1])} confounders and {len(labels[labels == 0])} non-confounders")

        group_label_file = os.path.join(self.base_dir, f"group_labels_attrbs-layer-{self.attribution_layer}.json")
        with open(group_label_file, 'w', encoding='utf-8') as file:
            json.dump(group_label_map, file)

        self.train_data.enable_groups()
        self.val_data.enable_groups()
        self.test_data.enable_groups()
        dataloader = DataLoader(ConcatDataset([self.train_data, self.val_data, self.test_data]), batch_size=64, shuffle=False)
        label_acc = []
        for batch in dataloader:
            labels = batch["has_confounder"]
            for i, filename in enumerate(batch['url']):
                if filename in group_label_map:
                    label_acc.append(labels[i].item() == group_label_map[filename])
        print(f"{np.mean(label_acc) * 100}% of {len(label_acc)} labels correct!")

        return group_label_map


    def _compute_attributions(self, attribution_layer: str):

        self.attribution_db_path = os.path.join(self.analysis_dir, "attribution.h5")
        self.heatmaps_db_path = os.path.join(self.analysis_dir, "heatmaps.h5")
        self.concept_importance_db_path = os.path.join(self.analysis_dir, "concept_importance.h5")
        self.input_db_path = os.path.join(self.analysis_dir, "input.h5")

        if pathlib.Path(self.input_db_path).is_file():
            return

        attribution_db_file = None
        input_db_file = None
        heatmaps_db_file = None
        concept_importance_db_file = None

        attribution = CondAttribution(self.model)
        canonizer = ResNetCanonizer()
        composite = EpsilonPlusFlat([canonizer])
        cc = ChannelConcept()

        train_data = self.train_data
        if len(self.train_data) > self.config.max_samples:
            idxs = np.random.choice(np.arange(len(self.train_data)), size=self.config.max_samples, replace=False)
            train_data = Subset(self.train_data, idxs)
        dataloader = DataLoader(ConcatDataset([train_data, self.val_data, self.test_data]), batch_size=64, shuffle=False)

        number_samples_processed = 0
        with tqdm(dataloader) as pbar:
            pbar.set_description(f"computing relevances at layer {attribution_layer}...")
            for batch in pbar:
                x = batch["x"].to(self.device).requires_grad_()
                y = batch["y"].squeeze().int()
                out = self.model(x).detach().cpu()

                condition = [{"y": target} for target in y]
                # condition = [{"y": c_id} for c_id in out.argmax(1)]
                attr = attribution(x, condition, composite, record_layer=[attribution_layer], init_rel=1)
                if self.config.skip_wrongly_classified_samples:
                    non_zero = ((attr.heatmap.sum((1, 2)).abs().detach().cpu() > 0) * (out.argmax(1) == y)).numpy()
                else:
                    non_zero = (attr.heatmap.sum((1, 2)).abs().detach().cpu() > 0).numpy()

                if len(non_zero) == 0:
                    continue

                attributions = attr.relevances[attribution_layer][non_zero]
                concept_importance = cc.attribute(attributions, abs_norm=True)[:, None]
                heatmaps = attr.heatmap[non_zero][:, None]
                y = y[non_zero]
                x = x[non_zero]

                if number_samples_processed == 0:
                    print("attribution shape:", attributions.shape)
                    number_samples = len(dataloader.dataset)
                    attribution_db_file = create_attribution_database(self.attribution_db_path, attributions[0].shape, self.config.classes_total, number_samples)
                    heatmaps_db_file = create_attribution_database(self.heatmaps_db_path, heatmaps[0].shape, self.config.classes_total, number_samples)
                    concept_importance_db_file = create_attribution_database(self.concept_importance_db_path, concept_importance[0].shape, self.config.classes_total, number_samples)
                    input_db_file = create_input_database(self.input_db_path, x[0].shape, len(dataloader.dataset))

                append_attributions(attribution_db_file, number_samples_processed, attributions, attr.prediction[non_zero], y)
                append_attributions(heatmaps_db_file, number_samples_processed, heatmaps, attr.prediction[non_zero], y)
                append_attributions(concept_importance_db_file, number_samples_processed, concept_importance, attr.prediction[non_zero], y)
                append_inputs(input_db_file, number_samples_processed, x.detach().cpu(), y, np.asarray(batch["url"])[non_zero])
                number_samples_processed += len(y)

        print(f"{number_samples_processed} of {len(dataloader.dataset)} processed")
        resize_input_db(input_db_file, number_samples_processed)
        resize_attribution_db(attribution_db_file, number_samples_processed)
        resize_attribution_db(heatmaps_db_file, number_samples_processed)
        resize_attribution_db(concept_importance_db_file, number_samples_processed)
        attribution_db_file.close()
        heatmaps_db_file.close()
        concept_importance_db_file.close()
        input_db_file.close()

    def _spectral_clustering(self):

        analysis_db_path = os.path.join(self.analysis_dir, f"analysis.h5")
        if pathlib.Path(analysis_db_path).is_file():
            return analysis_db_path

        number_of_clusters_list = list(range(2, self.config.num_clusters_max + 1))
        pipeline = SpectralClustering(
            preprocessing=Sequential(self._get_preprocessing_pipeline()),
            embedding=EigenDecomposition(n_eigval=self.config.num_eigval, is_output=True),
            clustering=Parallel([
                Parallel([
                    KMeans(n_clusters=number_of_clusters, kwargs={"max_iter": 700}) for number_of_clusters in number_of_clusters_list
                ], broadcast=True),
                Parallel([
                    AgglomerativeClustering(n_clusters=number_of_clusters) for number_of_clusters in number_of_clusters_list
                ], broadcast=True),
                DBSCAN(),
                HDBSCAN(),
                TSNEEmbedding(perplexity=self.config.tsne_perplexity, kwargs={"learning_rate": "auto", "init": "pca"}),
                UMAPEmbedding(n_neighbors=self.config.umap_neighbors)
            ], broadcast=True, is_output=True)
        )

        attribution_path = self.concept_importance_db_path if self.config.use_relative_concept_importance else self.attribution_db_path
        with h5py.File(attribution_path, 'r') as attributions_file:
            labels = attributions_file['label'][:]

        for class_idx in range(self.config.classes_total):

            with h5py.File(attribution_path, 'r') as attributions_file:
                indices_of_samples_in_class, = np.nonzero(labels == class_idx)
                attribution_data = attributions_file['attribution'][indices_of_samples_in_class, :]
                print(f"len attribution data for class {class_idx}: {len(attribution_data)}")

            (eigenvalues, embedding), (kmeans, ac, dbscan, hdbscan, tsne, umap) = pipeline(attribution_data)

            with h5py.File(analysis_db_path, 'a') as analysis_file:

                analysis_name = CLASS_NAMES[self.dataset_class][class_idx]

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

                cluster_group['DBSCAN'] = dbscan
                cluster_group['DBSCAN'].attrs['embedding'] = 'spectral'
                cluster_group['DBSCAN'].attrs['index'] = np.arange(embedding.shape[1], dtype=np.uint32)

                cluster_group['HDBSCAN'] = hdbscan
                cluster_group['HDBSCAN'].attrs['embedding'] = 'spectral'
                cluster_group['HDBSCAN'].attrs['index'] = np.arange(embedding.shape[1], dtype=np.uint32)

        return analysis_db_path

    def _create_label_map(self) -> str:
        label_map_file_path = os.path.join(self.analysis_dir, "label_map.json")
        if pathlib.Path(label_map_file_path).is_file():
            return label_map_file_path

        class_names = CLASS_NAMES[self.dataset_class]
        label_map = [{
            'index': idx,
            'name': class_names[idx],
            'word_net_id': str(idx)
        } for idx in range(self.config.classes_total)]
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

        group_label_dir = os.path.join(self.analysis_dir, "group-labels")

        if not os.path.exists(group_label_dir):
            # exit()
            host_name = "0.0.0.0"
            workspace = Workspace()
            workspace.add_project(self.virelay_project_path)
            app = Server(workspace)
            proc = multiprocessing.Process(
                target=lambda: app.run(host=host_name, port=self.config.port), args=())
            proc.start()
            print('ViReLay GUI is active on localhost:' + str(self.config.port))

            print(f'For each class, select the samples containing the confounding feature and export, '
                  f'then do the same for the samples without the confounding feature. Rename the files to '
                  f'"<cls-idx>-confounders.json" and "<cls-idx>-nonconfounders.json", respectively, and place '
                  f'them in the directory {group_label_dir} (leaving out some datapoints completely is generally '
                  f'ok, but might make it harder to correctly assess avg/worst group accuracy later, especially '
                  f'if some groups only have very few samples).')

            while not os.path.exists(group_label_dir):
                time.sleep(4.0)

            proc.terminate()
            print("Re-run script after placing the files in order to process results")
            exit()

        confounder_ids = []
        non_confounder_ids = []
        for file in os.listdir(group_label_dir):
            file = os.path.join(group_label_dir, file)
            if "non" in file:
                non_confounder_ids.extend(json.load(open(file, 'r'))['selectedDataPointIndices'])
            else:
                confounder_ids.extend(json.load(open(file, 'r'))['selectedDataPointIndices'])

        return confounder_ids, non_confounder_ids

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
        maxshape=(None,) + tuple(samples_shape),
        dtype='float32'
    )
    dataset_file.create_dataset(
        'label',
        shape=(number_of_samples,),
        maxshape=(None,),
        dtype='uint16'
    )
    dataset_file.create_dataset(
        'filenames',
        shape=(number_of_samples,),
        maxshape=(None,),
        dtype=h5py.string_dtype()
    )
    return dataset_file

def append_inputs(dataset_file, index, sample, label, filenames):
    dataset_file['data'][index:sample.shape[0] + index] = sample
    dataset_file['label'][index:label.shape[0] + index] = label
    dataset_file['filenames'][index:len(filenames) + index] = filenames

def resize_input_db(database_file, new_size):
    database_file['data'].resize(new_size, axis=0)
    database_file['label'].resize(new_size, axis=0)
    database_file['filenames'].resize(new_size, axis=0)

def create_attribution_database(attribution_database_file_path,
                                attribution_shape,
                                number_of_classes,
                                number_of_samples):

    attribution_database_file = h5py.File(attribution_database_file_path, 'w')
    attribution_database_file.create_dataset(
        'attribution',
        shape=(number_of_samples,) + tuple(attribution_shape),
        maxshape=(None,) + tuple(attribution_shape),
        dtype='float32'
    )
    attribution_database_file.create_dataset(
        'prediction',
        shape=(number_of_samples, number_of_classes),
        maxshape=(None, number_of_classes),
        dtype='float32'
    )
    attribution_database_file.create_dataset(
        'label',
        shape=(number_of_samples,),
        maxshape=(None,),
        dtype='uint16'
    )
    return attribution_database_file

def append_attributions(attribution_database_file,
                        index,
                        attributions,
                        predictions,
                        labels):

    attribution_database_file['attribution'][index:attributions.shape[0] + index] = attributions.detach().cpu().numpy()
    attribution_database_file['prediction'][index:predictions.shape[0] + index] = predictions.detach().cpu().numpy()
    attribution_database_file['label'][index:labels.shape[0] + index] = labels.detach().cpu().numpy()

def resize_attribution_db(database_file, new_size):
    database_file['attribution'].resize(new_size, axis=0)
    database_file['prediction'].resize(new_size, axis=0)
    database_file['label'].resize(new_size, axis=0)

CLASS_NAMES = {
    "SquareDataset": ['0 - Dark Foreground', '1 - Bright Foreground'],
    "ColoredMnist": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "Follicles": ['0', '1'],
    "CelebaMale": ["0 - Female", "1 - Male"],
    "CelebaSmiling": ["0 - Not Smiling", "1 - Smiling"],
    "CelebaBlond": ["0 - Not Blond", "1 - Blond"],
    "WaterbirdsDataset": ["0 - Landbird", "1 - Waterbird"],
    "Camelyon17AugmentedDataset": ["0 - No Tumor", "1 - Tumor"]
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
