import json
import multiprocessing
import os
import pathlib
import time

import h5py
import numpy as np
import torch
import yaml
from corelay.pipeline.spectral import SpectralClustering
from corelay.processor.base import Processor
from corelay.processor.clustering import KMeans, AgglomerativeClustering
from corelay.processor.embedding import EigenDecomposition, TSNEEmbedding, UMAPEmbedding
from corelay.processor.flow import Sequential, Parallel
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from tqdm import tqdm
from virelay.model import Workspace
from virelay.server import Server
from zennit.composites import EpsilonPlusFlat
from zennit.torchvision import ResNetCanonizer


# custom processors can be implemented by defining a function attribute
class Flatten(Processor):
    def function(self, data):
        return data.reshape(data.shape[0], np.prod(data.shape[1:]))


class Normalize(Processor):
    def function(self, data):
        print("data shape", data.shape)
        data = data / data.sum((1, 2, 3), keepdims=True)
        return data

def spectral_clustering(attribution_db_path: str, analysis_db_path: str, dataset_name: str, class_index: int):

    if pathlib.Path(analysis_db_path).is_file():
        return

    number_of_clusters_list = list(range(2, 10))
    pipeline = SpectralClustering(
        preprocessing=Sequential([Normalize(), Flatten()]),
        embedding=EigenDecomposition(n_eigval=32, is_output=True),
        clustering=Parallel([
            Parallel([
                KMeans(n_clusters=number_of_clusters, kwargs={"max_iter": 400}) for number_of_clusters in number_of_clusters_list
            ], broadcast=True),
            Parallel([
                AgglomerativeClustering(n_clusters=number_of_clusters) for number_of_clusters in number_of_clusters_list
            ], broadcast=True),
            TSNEEmbedding(),
            UMAPEmbedding()
        ], broadcast=True, is_output=True)
    )

    class_name = CLASS_NAMES[dataset_name][class_index]

    with h5py.File(attribution_db_path, 'r') as attributions_file:
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


def compute_attributions(model, dataloader, record_layer, attacked_class, analysis_dir, device):

    attribution_db_path = os.path.join(analysis_dir, "attribution.h5")
    heatmaps_db_path = os.path.join(analysis_dir, "heatmaps.h5")
    concept_importance_db_path = os.path.join(analysis_dir, "concept_importance.h5")
    input_db_path = os.path.join(analysis_dir, "input.h5")

    if pathlib.Path(attribution_db_path).is_file():
        return input_db_path, attribution_db_path, heatmaps_db_path, concept_importance_db_path

    attribution_db_file = None
    input_db_file = None
    heatmaps_db_file = None
    concept_importance_db_file = None

    attribution = CondAttribution(model)
    canonizer = ResNetCanonizer()
    composite = EpsilonPlusFlat([canonizer])
    cc = ChannelConcept()

    number_samples_processed = 0
    with tqdm(dataloader) as pbar:
        pbar.set_description(f"calculating activations and relevances at layer {record_layer}...")
        for batch in pbar:
            x = batch["x"].to(device).requires_grad_()
            y = batch["y"]
            if len(y.shape) == 2:
                y = y[:, 0]

            condition = [{"y": int(attacked_class)}] * len(x)
            attr = attribution(x, condition, composite, record_layer=[record_layer], init_rel=1)

            activation = attr.activations[record_layer].detach()
            relevance = attr.relevances[record_layer]
            concept_importance = cc.attribute(relevance)[:, None, None]
            heatmaps = attr.heatmap[:, None]

            if number_samples_processed == 0:
                number_classes = attr.prediction.shape[1]
                number_samples = len(dataloader.dataset)
                attribution_db_file = create_attribution_database(attribution_db_path, relevance[0].shape, number_classes, number_samples)
                heatmaps_db_file = create_attribution_database(heatmaps_db_path, heatmaps[0].shape, number_classes, number_samples)
                concept_importance_db_file = create_attribution_database(concept_importance_db_path, concept_importance[0].shape, number_classes, number_samples)
                input_db_file = create_input_database(input_db_path, x[0].shape, activation[0].shape, len(dataloader.dataset))

            append_attributions(attribution_db_file, number_samples_processed, relevance, attr.prediction, y)
            append_attributions(heatmaps_db_file, number_samples_processed, heatmaps, attr.prediction, y)
            append_attributions(concept_importance_db_file, number_samples_processed, concept_importance, attr.prediction, y)
            append_inputs(input_db_file, number_samples_processed, x.detach().cpu(), activation.detach().cpu(), y)
            number_samples_processed += len(y)

    attribution_db_file.close()
    heatmaps_db_file.close()
    concept_importance_db_file.close()
    input_db_file.close()

    return input_db_path, attribution_db_path, heatmaps_db_path, concept_importance_db_path

def make_project(analysis_dir: str,
                 dataset_file_path: str,
                 attribution_database_file_path: str,
                 analysis_file_path: str,
                 label_map_file_path: str,
                 dataset_name: str,
                 input_size: tuple[int]) -> str:

    output_file_path = os.path.join(analysis_dir, "virelay_project.yml")
    if pathlib.Path(output_file_path).is_file():
        return output_file_path

    project = {
        'project': {
            'name': "Finding Confounding Factors",
            'model': "PyTorch ResNet-18",
            'label_map': os.path.split(label_map_file_path)[1],
            'dataset': {
                'name': dataset_name,
                'type': 'hdf5',
                'path': os.path.split(dataset_file_path)[1],
                'input_width': input_size[1],
                'input_height': input_size[2],
                'down_sampling_method': None,
                'up_sampling_method': None,
            },
            'attributions': {
                'attribution_method': "Conditional Attribution EpsilonPlusFlat",
                'attribution_strategy': 'true_label',
                'sources': [os.path.split(attribution_database_file_path)[1]],
            },
            'analyses': [
                {
                    'analysis_method': "Spectral",
                    'sources': [os.path.split(analysis_file_path)[1]],
                }
            ]
        }
    }

    with open(output_file_path, 'w', encoding='utf-8') as project_file:
        yaml.dump(project, project_file, default_flow_style=False)

    return output_file_path

def run_virelay(analysis_dir, project_path: str, port: int = 8080) -> torch.Tensor:
    output_path = os.path.join(analysis_dir, "feedback.json")

    host_name = "0.0.0.0"
    workspace = Workspace()
    workspace.add_project(project_path)
    app = Server(workspace)
    proc = multiprocessing.Process(
        target=lambda: app.run(host=host_name, port=port), args=())
    proc.start()
    print('ViReLay GUI is active on localhost:' + str(port))
    print(f'Give feedback at localhost:{port} and save under {output_path} when done.')

    while not os.path.exists(output_path):
        time.sleep(5.0)

    proc.terminate()
    feedback = json.load(open(output_path, 'r'))

    return feedback['selectedDataPointIndices']

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

def create_input_database(dataset_file_path, samples_shape, activations_shape, number_of_samples):

    dataset_file = h5py.File(dataset_file_path, 'w')
    dataset_file.create_dataset(
        'data',
        shape=(number_of_samples,) + tuple(samples_shape),
        dtype='float32'
    )
    dataset_file.create_dataset(
        'activation',
        shape=(number_of_samples,) + tuple(activations_shape),
        dtype='float32'
    )
    dataset_file.create_dataset(
        'label',
        shape=(number_of_samples,),
        dtype='uint16'
    )
    return dataset_file

def append_inputs(dataset_file, index, sample, activation, label):
    dataset_file['data'][index:sample.shape[0] + index] = sample
    dataset_file['activation'][index:activation.shape[0] + index] = activation
    dataset_file['label'][index:label.shape[0] + index] = label

def create_label_map_file(analysis_dir, number_of_classes, dataset_name) -> str:
    label_map_file_path = os.path.join(analysis_dir, "label_map.json")
    if pathlib.Path(label_map_file_path).is_file():
        return label_map_file_path

    class_names = CLASS_NAMES[dataset_name]
    label_map = [{
        'index': index,
        'name': class_names[index],
        'word_net_id': str(index)
    } for index in range(number_of_classes)]
    with open(label_map_file_path, 'w', encoding='utf-8') as label_map_file:
        json.dump(label_map, label_map_file)

    return label_map_file_path

CLASS_NAMES = {
    "SquareDataset": ['Low Foreground Intensity','High Foreground Intensity'],
    "MnistDataset": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
}