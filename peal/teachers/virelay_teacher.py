import shutil

import torch
import json
import h5py
import yaml
import os
import time
import multiprocessing
import numpy as np
import socket

from tqdm import tqdm
from os.path import dirname, relpath
from typing import List
from corelay.base import Param
from corelay.processor.base import Processor
from corelay.processor.affinity import SparseKNN
from corelay.processor.distance import SciPyPDist
from corelay.processor.flow import Sequential, Parallel
from corelay.pipeline.spectral import SpectralClustering
from corelay.processor.embedding import TSNEEmbedding, UMAPEmbedding, EigenDecomposition
from corelay.processor.clustering import KMeans, DBSCAN, AgglomerativeClustering
from virelay.server import Server
from virelay.model import Workspace
from PIL import Image
from torchvision.transforms import ToTensor
from itertools import product
from pathlib import Path

from peal.global_utils import load_yaml_config, is_port_in_use, get_intermediate_output
from peal.teachers.interfaces import TeacherInterface


class ImageDataset(torch.utils.data.Dataset):
    """Shape Attribute dataset."""

    def __init__(self, root_dir, transform=ToTensor()):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.urls = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        """ """
        file = self.urls[idx]
        img = Image.open(os.path.join(self.root_dir, file))
        img = self.transform(img)[:3]

        return img, torch.tensor(0.0)


def create_dataset(
    dataset_file_path: str,
    samples_shape: tuple,
    number_of_samples: int,
    dataloader: torch.utils.data.DataLoader,
) -> h5py.File:
    """Creates a new dataset HDF5 file.
    Parameters
    ----------
        dataset_file_path: str
            The path to the dataset HDF5 file that is to be created.
        sample_shape: torch.Size
            The shape of the samples in the dataset.
        number_of_samples: int
            The number of samples in the dataset.
    Returns
    -------
        h5py.File
            Returns the file handle to the attributions database.
    """

    dataset_file = h5py.File(dataset_file_path, "w")
    dataset_file.create_dataset(
        "data", shape=(number_of_samples,) + tuple(samples_shape), dtype="float32"
    )
    dataset_file.create_dataset("label", shape=(number_of_samples,), dtype="uint16")
    index = 0
    for X, y in dataloader:
        dataset_file["data"][index : X.shape[0] + index] = X.detach().numpy()
        dataset_file["label"][index : y.shape[0] + index] = y.detach().numpy()
        index += X.shape[0]

    return dataset_file


def create_attribution_database(
    attribution_database_file_path, attribution_shape, num_classes, number_of_samples
):
    """
    Creates an attribution database file
    """
    attribution_database_file = h5py.File(attribution_database_file_path, "w")
    attribution_database_file.create_dataset(
        "attribution",
        shape=(number_of_samples,) + tuple(attribution_shape),
        dtype="float32",
    )
    attribution_database_file.create_dataset(
        "prediction", shape=(number_of_samples, num_classes), dtype="float32"
    )
    attribution_database_file.create_dataset(
        "label", shape=(number_of_samples,), dtype="uint16"
    )
    return attribution_database_file


def append_attributions(
    attribution_database_file, index, attributions, predictions, labels
):
    """
    Appends attributions to the attribution database file
    """
    attribution_database_file["attribution"][
        index : attributions.shape[0] + index
    ] = attributions.detach().numpy()
    attribution_database_file["prediction"][
        index : predictions.shape[0] + index
    ] = predictions.detach().numpy()
    attribution_database_file["label"][
        index : labels.shape[0] + index
    ] = labels.detach().numpy()


def create_attribution_dataset(dataloader, attribution_database_file_path, num_classes):
    """ """
    number_of_samples = len(dataloader)
    with create_attribution_database(
        attribution_database_file_path=attribution_database_file_path,
        attribution_shape=dataloader[0][0].shape,
        num_classes=num_classes,
        number_of_samples=number_of_samples,
    ) as attribution_database_file:
        number_of_samples_processed = 0
        for attributions, predictions, labels in dataloader:
            append_attributions(
                attribution_database_file=attribution_database_file,
                index=number_of_samples_processed,
                attributions=attributions.unsqueeze(0),
                predictions=torch.tensor([predictions]),
                labels=torch.tensor([labels]),
            )
            number_of_samples_processed += 1


class Flatten(Processor):
    """Represents a CoRelAy processor, which flattens its input data."""

    def function(self, data: np.ndarray) -> np.ndarray:
        """Applies the flattening to the input data.
        Parameters
        ----------
            data: np.ndarray
                The input data that is to be flattened.
        Returns
        -------
            np.ndarray
                Returns the flattened data.
        """

        return data.reshape(data.shape[0], np.prod(data.shape[1:]))


class SumChannel(Processor):
    """Represents a CoRelAy processor, which sums its input data across channels, i.e., its second axis."""

    def function(self, data: np.ndarray) -> np.ndarray:
        """Applies the summation over the channels to the input data.
        Parameters
        ----------
            data: np.ndarray
                The input data that is to be summed over its channels.
        Returns
        -------
            np.ndarray
                Returns the data that was summed up over its channels.
        """

        return data.sum(axis=1)


class Absolute(Processor):
    """Represents a CoRelAy processor, which computes the absolute value of its input data."""

    def function(self, data: np.ndarray) -> np.ndarray:
        """Computes the absolute value of the specified input data.
        Parameters
        ----------
            data: np.ndarray
                The input data for which the absolute value is to be computed.
        Returns
        -------
            np.ndarray
                Returns the absolute value of the input data.
        """

        return np.absolute(data)


class Normalize(Processor):
    """Represents a CoRelAy processor, which normalizes its input data.
    Attributes
    ----------
        axes: Param
            A parameter of the processor, which determines the axis over which the data is to be normalized. Defaults to
            the second and third axes.
    """

    axes = Param(tuple, (1, 2))

    def function(self, data: np.ndarray) -> np.ndarray:
        """Normalizes the specified input data.
        Parameters
        ----------
            data: np.ndarray
                The input data that is to be normalized.
        Returns
        -------
            np.ndarray
                Returns the normalized input data.
        """

        return data / data.sum(self.axes, keepdims=True)


class Histogram(Processor):
    """Represents a CoRelAy processor, which computes a histogram over its input data.
    Attributes
    ----------
        bins: Param
            A parameter of the processor, which determines the number of bins that are used to compute the histogram.
    """

    bins = Param(int, 256)

    def function(self, data: np.ndarray) -> np.ndarray:
        """Computes histograms over the specified input data. One histogram is computed for each channel and each sample
        in a batch of input data.
        Parameters
        ----------
            data: np.ndarray
                The input data over which the histograms are to be computed.
        Returns
        -------
            np.ndarray
                Returns the histograms that were computed over the input data.
        """

        return np.stack(
            [
                np.stack(
                    [
                        np.histogram(
                            sample.reshape(sample.shape[0], np.prod(sample.shape[1:3])),
                            bins=self.bins,
                            density=True,
                        )
                        for sample in channel
                    ]
                )
                for channel in data.transpose(3, 0, 1, 2)
            ]
        )


# Contains the various pre-processing method and distance metric variants that can be used to compute the analysis
VARIANTS = {
    "absspectral": {
        "preprocessing": Sequential([Absolute(), SumChannel(), Normalize(), Flatten()]),
        "distance": SciPyPDist(metric="euclidean"),
    },
    "spectral": {
        "preprocessing": Sequential([SumChannel(), Normalize(), Flatten()]),
        "distance": SciPyPDist(metric="euclidean"),
    },
    "fullspectral": {
        "preprocessing": Sequential([Normalize(axes=(1, 2, 3)), Flatten()]),
        "distance": SciPyPDist(metric="euclidean"),
    },
    "histogram": {
        "preprocessing": Sequential(
            [Normalize(axes=(1, 2, 3)), Histogram(), Flatten()]
        ),
        "distance": SciPyPDist(metric="euclidean"),
    },
}


def meta_analysis(
    base_dir: str,
    attribution_database_file_path: str,
    analysis_file_path: str,
    variant: str,
    class_indices: List[int],
    label_map_file_path: str,
    number_of_eigenvalues: int,
    number_of_clusters_list: List[int],
    number_of_neighbors: int,
) -> None:
    """Performs a meta-analysis over the specified attribution data and writes the results into an analysis database.
    Parameters
    ----------
        base_dir: str
            The base path to the CoRelAy project.
        attribution_database_file_path: str
            The path to the attribution database file, that contains the attributions for which the meta-analysis is to
            be performed.
        analysis_file_path: str
            The path to the analysis database file, into which the results of the meta-analysis are to be written.
        variant: str
            The meta-analysis variant that is to be performed. Can be one of "absspectral", "spectral", "fullspectral",
            or "histogram".
        class_indices: List[int]
            The indices of the classes for which the meta-analysis is to be performed. If not specified, then the
            meta-analysis is performed for all classes.
        label_map_file_path: str
            The path to the label map file, which contains a mapping between the class indices and their corresponding
            names and WordNet IDs.
        number_of_eigenvalues: int
            The number of eigenvalues of the eigenvalue decomposition.
        number_of_clusters_list: List[int]
            A list that can contain multiple numbers of clusters. For each number of clusters in this list, all
            clustering methods and the meta-analysis are performed.
        number_of_neighbors: int
            The number of neighbors that are to be considered in the k-nearest neighbor clustering algorithm.
    """
    # Determines the pre-processing pipeline and the distance metric that are to be used for the meta-analysis
    pre_processing_pipeline = VARIANTS[variant]["preprocessing"]
    distance_metric = VARIANTS[variant]["distance"]

    # Creates the meta-analysis pipeline
    """pipeline = SpectralClustering(
        preprocessing=pre_processing_pipeline,
        pairwise_distance=distance_metric,
        affinity=SparseKNN(n_neighbors=number_of_neighbors, symmetric=True),
        embedding=EigenDecomposition(
            n_eigval=number_of_eigenvalues, is_output=True),
        clustering=Parallel([
            Parallel([
                KMeans(n_clusters=number_of_clusters) for number_of_clusters in number_of_clusters_list
            ], broadcast=True),
            Parallel([
                DBSCAN(eps=number_of_clusters / 10.0) for number_of_clusters in number_of_clusters_list
            ], broadcast=True),
            Parallel([
                AgglomerativeClustering(n_clusters=number_of_clusters) for number_of_clusters in number_of_clusters_list
            ], broadcast=True),
            Parallel([
                UMAPEmbedding(),
                TSNEEmbedding(),
            ], broadcast=True)
        ], broadcast=True, is_output=True)
    )"""
    pipeline = SpectralClustering(
        preprocessing=pre_processing_pipeline,
        pairwise_distance=distance_metric,
        affinity=SparseKNN(n_neighbors=number_of_neighbors, symmetric=True),
        embedding=EigenDecomposition(n_eigval=2, is_output=True),
        clustering=Parallel(
            [
                Parallel(
                    [
                        KMeans(n_clusters=number_of_clusters)
                        for number_of_clusters in number_of_clusters_list
                    ],
                    broadcast=True,
                ),
                Parallel(
                    [
                        DBSCAN(eps=number_of_clusters / 10.0)
                        for number_of_clusters in number_of_clusters_list
                    ],
                    broadcast=True,
                ),
                Parallel(
                    [
                        AgglomerativeClustering(n_clusters=number_of_clusters)
                        for number_of_clusters in number_of_clusters_list
                    ],
                    broadcast=True,
                ),
                Parallel(
                    [
                        Parallel(
                            [
                                UMAPEmbedding(
                                    n_neighbors=number_of_neighbors, metric="cosine"
                                )
                                for number_of_neighbors in [5, 10, 15, 20, 30]
                            ],
                            broadcast=True,
                        ),
                        Parallel(
                            [
                                TSNEEmbedding(
                                    perplexity=float(perp),
                                    early_exaggeration=float(exagg),
                                    metric="cosine",
                                )
                                for perp, exagg in product(
                                    [5, 15, 30, 50, 100], [6, 12, 24, 32, 48]
                                )
                            ],
                            broadcast=True,
                        ),
                    ],
                    broadcast=True,
                ),
            ],
            broadcast=True,
            is_output=True,
        ),
    )

    # Loads the label map and converts it to a dictionary, which maps the class index to its WordNet ID
    if label_map_file_path is not None:
        with open(label_map_file_path, "r", encoding="utf-8") as label_map_file:
            label_map = json.load(label_map_file)
        wordnet_id_map = {label["index"]: label["word_net_id"] for label in label_map}
        class_name_map = {label["index"]: label["name"] for label in label_map}

    else:
        wordnet_id_map = {
            class_index: str(class_index) for class_index in class_indices
        }
        class_name_map = {
            class_index: str(class_index) for class_index in class_indices
        }
        label_map = [
            {
                "index": class_index,
                "word_net_id": str(class_index),
                "name": str(class_index),
            }
            for class_index in class_indices
        ]
        with open(
            os.path.join(base_dir, "label_map.json"), "w", encoding="utf-8"
        ) as label_map_file:
            json.dump(label_map, label_map_file, indent=4)

    # Retrieves the labels of the samples
    with h5py.File(attribution_database_file_path, "r") as attributions_file:
        labels = attributions_file["label"][:]

    # Gets the indices of the classes for which the meta-analysis is to be performed, if non were specified, the the
    # meta-analysis is performed for all classes
    if class_indices is None:
        class_indices = [int(label["index"]) for label in label_map]

    # Truncate the analysis database
    print(f"Truncating {analysis_file_path}")
    h5py.File(analysis_file_path, "w").close()

    # Cycles through all classes and performs the meta-analysis for each of them
    for class_index in class_indices:
        # Loads the attribution data for the samples of the current class
        print(f"Loading class {class_name_map[class_index]}")
        with h5py.File(attribution_database_file_path, "r") as attributions_file:
            (indices_of_samples_in_class,) = np.nonzero(labels == class_index)
            attribution_data = attributions_file["attribution"][
                indices_of_samples_in_class, :
            ]
            print("Samples in class: " + str(attribution_data.shape[0]))
            if "train" in attributions_file:
                train_flag = attributions_file["train"][
                    indices_of_samples_in_class.tolist()
                ]
            else:
                train_flag = None

        # Performs the meta-analysis for the attributions of the current class
        print(f"Computing class {class_name_map[class_index]}")
        (eigenvalues, embedding), (
            kmeans,
            dbscan,
            agglomerative,
            (umap, tsne),
        ) = pipeline(attribution_data)

        # Append the meta-analysis to the analysis database
        print(f"Saving class {class_name_map[class_index]}")
        with h5py.File(analysis_file_path, "a") as analysis_file:
            # The name of the analysis is the name of the class
            analysis_name = wordnet_id_map.get(class_index, f"{class_index:08d}")

            # Adds the indices of the samples in the current class to the analysis database
            analysis_group = analysis_file.require_group(analysis_name)
            analysis_group["index"] = indices_of_samples_in_class.astype("uint32")

            # Adds the spectral embedding to the analysis database
            embedding_group = analysis_group.require_group("embedding")
            embedding_group["spectral"] = embedding.astype(np.float32)
            embedding_group["spectral"].attrs["eigenvalue"] = eigenvalues.astype(
                np.float32
            )

            # Adds the t-SNE embedding to the analysis database
            for tsne_idx, tsne_embedding in enumerate(tsne):
                embedding_name = f"tsne-{tsne_idx:02d}"
                embedding_group[embedding_name] = tsne_embedding.astype(np.float32)
                embedding_group[embedding_name].attrs["embedding"] = "spectral"
                embedding_group[embedding_name].attrs["k"] = tsne_idx
                embedding_group[embedding_name].attrs["index"] = np.array([0, 1])

            # Adds the t-SNE embedding to the analysis database
            for umap_idx, umap_embedding in enumerate(umap):
                embedding_name = f"umap-{umap_idx:02d}"
                embedding_group[embedding_name] = umap_embedding.astype(np.float32)
                embedding_group[embedding_name].attrs["embedding"] = "spectral"
                embedding_group[embedding_name].attrs["k"] = umap_idx
                embedding_group[embedding_name].attrs["index"] = np.array([0, 1])

            # Adds the k-means clustering of the embeddings to the analysis database
            cluster_group = analysis_group.require_group("cluster")
            for number_of_clusters, clustering in zip(number_of_clusters_list, kmeans):
                clustering_dataset_name = f"kmeans-{number_of_clusters:02d}"
                cluster_group[clustering_dataset_name] = clustering
                cluster_group[clustering_dataset_name].attrs["embedding"] = "spectral"
                cluster_group[clustering_dataset_name].attrs["k"] = number_of_clusters
                cluster_group[clustering_dataset_name].attrs["index"] = np.arange(
                    embedding.shape[1], dtype=np.uint32
                )

            # Adds the Agglomerative clustering of the embeddings to the analysis database
            for number_of_clusters, clustering in zip(
                number_of_clusters_list, agglomerative
            ):
                clustering_dataset_name = f"agglomerative-{number_of_clusters:02d}"
                cluster_group[clustering_dataset_name] = clustering
                cluster_group[clustering_dataset_name].attrs["embedding"] = "spectral"
                cluster_group[clustering_dataset_name].attrs["k"] = number_of_clusters
                cluster_group[clustering_dataset_name].attrs["index"] = np.arange(
                    embedding.shape[1], dtype=np.uint32
                )

            # If the attributions were computed on the training split of the dataset, then the training flag is set
            if train_flag is not None:
                cluster_group["train_split"] = train_flag


def make_project(
    dataset_file_path: str,
    attribution_database_file_path: str,
    analysis_file_path: str,
    label_map_file_path: str,
    project_name: str,
    dataset_name: str,
    dataset_down_sampling_method: str,
    dataset_up_sampling_method: str,
    model_name: str,
    attribution_name: str,
    analysis_name: str,
    output_file_path: str,
) -> None:
    """Generates a ViRelAy project file.
    Parameters
    ----------
        dataset_file_path: str
            The path to the dataset HDF5 file.
        attribution_database_file_path: str
            The path to the attribution HDF5 file.
        analysis_file_path: str
            The path to the analysis HDF5 file.
        label_map_file_path: str
            The path to the label map YAML file.
        project_name: str
            The name of the project.
        dataset_name: str
            The name of the dataset that the classifier was trained on.
        dataset_down_sampling_method: str
            The method that is to be used to down-sample images from the dataset that are larger than the input to the
            model. Must be one of "none", "center_crop", or "resize".
        dataset_up_sampling_method: str
            The method that is to be used to up-sample images from the dataset that are smaller than the input to the
            model. Must be one of "none", "fill_zeros", "fill_ones", "edge_repeat", "mirror_edge", "wrap_around", or
            "resize".
        model_name: str
            The name of the classifier model on which the project is based.
        attribution_name: str
            The name of the method that was used to compute the attributions.
        analysis_name: str
            The name of the analysis that was performed on the attributions.
        output_file_path: str
            The path to the YAML file into which the project will be saved.
    """

    # Determines the root path of the project, which is needed to make all paths stored in the project file relative to
    # the project file
    if output_file_path is not None:
        project_root_path = dirname(output_file_path)
    else:
        project_root_path = "."

    if label_map_file_path is None:
        label_map_file_path = os.path.join(project_root_path, "label_map.json")

    # Determines the shape of the dataset samples
    with h5py.File(dataset_file_path, "r") as dataset_file:
        input_shape = dataset_file["data"].shape  # pylint: disable=no-member

    # Generates the information, which will be stored in the project file
    project = {
        "project": {
            "name": project_name,
            "model": model_name,
            "label_map": relpath(label_map_file_path, start=project_root_path),
            "dataset": {
                "name": dataset_name,
                "type": "hdf5",
                "path": relpath(dataset_file_path, start=project_root_path),
                "input_width": input_shape[2],
                "input_height": input_shape[3],
                "down_sampling_method": dataset_down_sampling_method,
                "up_sampling_method": dataset_up_sampling_method,
            },
            "attributions": {
                "attribution_method": attribution_name,
                "attribution_strategy": "true_label",
                "sources": [
                    relpath(attribution_database_file_path, start=project_root_path)
                ],
            },
            "analyses": [
                {
                    "analysis_method": analysis_name,
                    "sources": [relpath(analysis_file_path, start=project_root_path)],
                }
            ],
        }
    }

    # If an output file path was specified, then the project is saved into the specified file, otherwise, the project
    # information is written to the standard output
    if output_file_path is None:
        print(yaml.dump(project, default_flow_style=False))
    else:
        with open(output_file_path, "w", encoding="utf-8") as project_file:
            yaml.dump(project, project_file, default_flow_style=False)


def run_virelay(project_path: str, output_path: str, port: int) -> None:
    host_name = "0.0.0.0"
    workspace = Workspace()
    workspace.add_project(project_path)
    app = Server(workspace)
    # thread = threading.Thread(target=lambda: app.run(host=host_name, port=port))
    # thread.start()
    proc = multiprocessing.Process(
        target=lambda: app.run(host=host_name, port=port), args=()
    )
    proc.start()
    print("ViReLay GUI is active on localhost:" + str(port))

    with tqdm(range(100000)) as pbar:
        for it in pbar:
            if os.path.exists(output_path):
                time.sleep(1.0)
                feedback = json.load(open(output_path, "r"))
                break

            else:
                pbar.set_description(
                    "Give feedback at localhost:"
                    + str(port)
                    + " and save under "
                    + output_path
                    + " when done."
                )
                time.sleep(1.0)

    proc.terminate()

    return feedback


class VirelayTeacher(TeacherInterface):
    def __init__(
        self,
        port=2000,
        dataset=None,
        tracking_level=5,
        num_classes=2,
        teacher_config="configs/cfkd_experiments/teachers/virelay_teacher.yaml",
        distance_from_last_layer=27,  # 15, #3, #39, #2, #50,
    ):
        self.port = port
        self.dataset = dataset
        self.tracking_level = tracking_level
        self.num_classes = num_classes
        self.teacher_config = load_yaml_config(teacher_config).__dict__
        self.distance_from_last_layer = distance_from_last_layer

    def get_feedback(
        self,
        collage_path_list,
        x_counterfactual_list,
        x_list,
        y_list,
        base_dir,
        student,
        y_source_list,
        y_target_end_confidence_list,
        **kwargs,
    ):
        # filter all lists to only contain only values where y_target_end_confidence_list is greater than 0.5
        # first create a list of indices where y_target_end_confidence_list is greater than 0.5
        y_target_confidence_greater_than_05_indices = [
            i
            for i in range(len(y_target_end_confidence_list))
            if y_target_end_confidence_list[i] > 0.5
        ]
        collage_path_greater_than_05_list = [
            collage_path_list[i]
            for i in range(len(y_target_end_confidence_list))
            if y_target_end_confidence_list[i] > 0.5
        ]
        x_counterfactual_greater_than_05_list = [
            x_counterfactual_list[i]
            for i in range(len(y_target_end_confidence_list))
            if y_target_end_confidence_list[i] > 0.5
        ]
        x_greater_than_05_list = [
            x_list[i]
            for i in range(len(y_target_end_confidence_list))
            if y_target_end_confidence_list[i] > 0.5
        ]
        """y_greater_than_05_list = [
            y_list[i] for i in range(len(y_target_end_confidence_list)) if y_target_end_confidence_list[i] > 0.5
        ]
        y_source_greater_than_05_list = [
            y_source_list[i] for i in range(len(y_target_end_confidence_list)) if y_target_end_confidence_list[i] > 0.5
        ]"""
        y_greater_than_05_list = [
            0
            for i in range(len(y_target_end_confidence_list))
            if y_target_end_confidence_list[i] > 0.5
        ]
        y_source_greater_than_05_list = [
            0
            for i in range(len(y_target_end_confidence_list))
            if y_target_end_confidence_list[i] > 0.5
        ]
        attributions = []
        for i in range(1, 1000):
            try:
                x_latent = get_intermediate_output(
                    student,
                    x_greater_than_05_list[0].unsqueeze(0).to("cuda"),
                    i,
                )
                print(str(i) + ": " + str(x_latent.shape))

            except Exception:
                break

        for i in range(len(x_greater_than_05_list)):
            x_latent = (
                get_intermediate_output(
                    student,
                    x_greater_than_05_list[i].unsqueeze(0).to("cuda"),
                    self.distance_from_last_layer,
                )
                .detach()
                .cpu()
            )
            cf_latent = (
                get_intermediate_output(
                    student,
                    x_counterfactual_greater_than_05_list[i].unsqueeze(0).to("cuda"),
                    self.distance_from_last_layer,
                )
                .detach()
                .cpu()
            )
            attribution_difference = cf_latent - x_latent
            # attribution_difference = attribution_difference.squeeze(0).flatten().unsqueeze(-1).unsqueeze(-1)
            rel = torch.sum(
                attribution_difference.view(*attribution_difference.shape[:2], -1),
                dim=-1,
            )
            rel = rel / (torch.abs(rel).sum(-1).view(-1, 1) + 1e-10)
            rel = rel.unsqueeze(-1).unsqueeze(-1)
            attributions.append(rel)

        print("attributions[0].shape")
        print("attributions[0].shape")
        print("attributions[0].shape")
        print(attributions[0].shape)
        shutil.rmtree(base_dir, ignore_errors=True)
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(os.path.join(base_dir, "database.hdf5")):
            # write collage_path_greater_than_05_list in the os.path.join(base_dir, "collage_greater_than_05") folder
            collage_path_greater_than_05_dir = os.path.join(
                base_dir, "collage_greater_than_05"
            )
            os.makedirs(collage_path_greater_than_05_dir, exist_ok=True)
            for i, collage_path in enumerate(collage_path_greater_than_05_list):
                # copy the collage_path to the collage_path_greater_than_05_dir
                os.system(
                    f"cp {collage_path} {collage_path_greater_than_05_dir}/{i}.jpg"
                )

            collage_path_list_dataloader = torch.utils.data.DataLoader(
                ImageDataset(collage_path_greater_than_05_dir),
            )
            create_dataset(
                dataset_file_path=os.path.join(base_dir, "database.hdf5"),
                samples_shape=collage_path_list_dataloader.dataset[0][0].shape,
                number_of_samples=len(collage_path_list_dataloader.dataset),
                dataloader=collage_path_list_dataloader,
            )

        # TODO get the actual prediction scores of the network
        if not os.path.exists(os.path.join(base_dir, "attribution_database.hdf5")):
            attributions_dataloader = list(
                zip(attributions, y_source_greater_than_05_list, y_greater_than_05_list)
            )
            create_attribution_dataset(
                attribution_database_file_path=os.path.join(
                    base_dir, "attribution_database.hdf5"
                ),
                dataloader=attributions_dataloader,
                num_classes=1,  # self.num_classes,
            )

        if not os.path.exists(os.path.join(base_dir, "analysis.hdf5")):
            meta_analysis(
                base_dir=base_dir,
                attribution_database_file_path=os.path.join(
                    base_dir, "attribution_database.hdf5"
                ),
                analysis_file_path=os.path.join(base_dir, "analysis.hdf5"),
                variant=self.teacher_config["meta_analysis_variant"],
                class_indices=range(1),  # range(self.num_classes),
                label_map_file_path=None,
                number_of_eigenvalues=self.teacher_config["number_of_eigenvalues"],
                number_of_clusters_list=self.teacher_config["number_of_clusters_list"],
                number_of_neighbors=self.teacher_config["number_of_neighbors"],
            )

        if not os.path.exists(os.path.join(base_dir, "analysis.yaml")):
            make_project(
                dataset_file_path=os.path.join(base_dir, "database.hdf5"),
                attribution_database_file_path=os.path.join(
                    base_dir, "attribution_database.hdf5"
                ),
                analysis_file_path=os.path.join(base_dir, "analysis.hdf5"),
                label_map_file_path=None,
                project_name="analysis",
                dataset_name="dataset",
                dataset_down_sampling_method=self.teacher_config[
                    "dataset_down_sampling_method"
                ],
                dataset_up_sampling_method=self.teacher_config[
                    "dataset_up_sampling_method"
                ],
                model_name="model",
                attribution_name="attribution",
                analysis_name="analysis",
                output_file_path=os.path.join(base_dir, "analysis.yaml"),
            )

        # TODO check if port is free
        if not os.path.exists(os.path.join(base_dir, "feedback.json")):
            while is_port_in_use(self.port):
                print("port " + str(self.port) + " is occupied!")
                self.port += 1

            run_virelay(
                project_path=os.path.join(base_dir, "analysis.yaml"),
                port=self.port,
                output_path=os.path.join(base_dir, "feedback.json"),
            )

        feedback = json.load(open(os.path.join(base_dir, "feedback.json"), "r"))

        # set the default feedback out to "Student not flipped" for the full length collage list
        feedback_out = ["Student not flipped" for _ in range(len(collage_path_list))]
        for i in range(len(collage_path_greater_than_05_list)):
            if i in feedback["selectedDataPointIndices"]:
                feedback_out[y_target_confidence_greater_than_05_indices[i]] = "false"

            else:
                feedback_out[y_target_confidence_greater_than_05_indices[i]] = "true"

        return feedback_out
