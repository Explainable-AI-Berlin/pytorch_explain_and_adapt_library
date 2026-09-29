import os
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm

from sae import load_model
from utils import SAEDataset, set_seed, get_device
from metrics import (
    explained_variance_full,
    normalized_mean_absolute_error,
    l0_messure,
    cknna
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the representation extraction and evaluation script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Extract and evaluate representations from Sparse Autoencoder models")
    parser.add_argument("-m", "--model", type=str, required=True, 
                        help="Path to the trained model file (.pt)")
    parser.add_argument("-d", "--data", type=str, required=True, 
                        help="Path to the dataset file (.npy)")
    parser.add_argument("-b", "--batch-size", type=int, default=10000, 
                        help="Batch size for processing data")
    parser.add_argument("-o", "--output-path", type=str, default=".", 
                        help="Directory path to save extracted representations")
    parser.add_argument("-s", "--seed", type=int, default=42, 
                        help="Random seed for reproducibility")

    return parser.parse_args()


def get_representation(model, dataset, repr_file_name, batch_size):
    """
    Extract representations from the model for the given dataset and evaluate model performance.
    
    Extracts both output reconstructions and latent representations from the model,
    saves them to disk as memory-mapped files, and computes various performance metrics.
    
    Args:
        model: The Sparse Autoencoder model to evaluate
        dataset: Dataset to process
        repr_file_name (str): Base filename for saving representations
        batch_size (int): Number of samples to process at once
        
    Metrics computed:
        - Fraction of Variance Unexplained (FVU) using normalized MSE
        - Normalized Mean Absolute Error (MAE)
        - Cosine similarity between inputs and outputs
        - L0 measure (average number of active neurons per sample)
        - CKNNA (Cumulative k-Nearest Neighbor Accuracy)
        - Number of dead neurons (neurons that never activate)
    """
    device = get_device()
    logger.info(f"Using device: {device}")
    
    model.eval()
    model.to(device)
    with torch.no_grad():
        # Prepare memory-mapped file for output reconstructions
        repr_file_name_output = f"{repr_file_name}_output_{len(dataset)}_{model.n_inputs}.npy"
        memmap_output = np.memmap(repr_file_name_output, dtype='float32', mode='w+', 
                                  shape=(len(dataset), model.n_inputs))
        logger.info(f"Data output with shape {memmap_output.shape} will be saved to {repr_file_name_output}")

        # Prepare memory-mapped file for latent representations
        repr_file_name_repr = f"{repr_file_name}_repr_{len(dataset)}_{model.n_latents}.npy"
        memmap_repr = np.memmap(repr_file_name_repr, dtype='float32', mode='w+', 
                                shape=(len(dataset), model.n_latents))
        logger.info(f"Data repr with shape {memmap_repr.shape} will be saved to {repr_file_name_repr}")

        # Create dataloader for batch processing
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                               shuffle=True, num_workers=0)
        
        # Lists to collect metrics for each batch
        l0 = []
        mae = []
        fvu = []
        cs = []
        cknnas = []
        sparse_l0 = []
        sparse_mae = []
        sparse_fvu = []
        sparse_cs = []
        sparse_cknnas = []
        dead_neurons_count = None
        
        # Process data in batches
        for idx, batch in enumerate(tqdm(dataloader, desc="Extracting representations")):
            start = batch_size * idx
            end = start + batch.shape[0]
            batch = batch.to(device)
            
            # Forward pass through the model
            with torch.no_grad():
                sparse_outputs, sparse_representation, outputs, representations = model(batch)
            
            # Post-process outputs and batch
            batch = dataset.unprocess_data(batch.cpu()).to(device)
            outputs = dataset.unprocess_data(outputs.cpu()).to(device)
            sparse_outputs = dataset.unprocess_data(sparse_outputs.cpu()).to(device)
            
            
            # Save the outputs and representations to the memmap files
            outputs_numpy = outputs.cpu().numpy()
            memmap_output[start:end] = outputs_numpy
            memmap_output.flush()

            representations_numpy = representations.cpu().numpy()
            memmap_repr[start:end] = representations_numpy
            memmap_repr.flush()

            # Calculate and collect metrics
            fvu.append(explained_variance_full(batch, outputs))
            mae.append(normalized_mean_absolute_error(batch, outputs))
            cs.append(torch.nn.functional.cosine_similarity(batch, outputs))
            l0.append(l0_messure(representations))
            # Only calculate the cknna if it even to the number of the batch
            if batch.shape[0] == batch_size:
                cknnas.append(cknna(batch, representations, topk=10))
            
            sparse_fvu.append(explained_variance_full(batch, sparse_outputs))
            sparse_mae.append(normalized_mean_absolute_error(batch, sparse_outputs))
            sparse_cs.append(torch.nn.functional.cosine_similarity(batch, sparse_outputs))
            sparse_l0.append(l0_messure(sparse_representation))
            # Only calculate the cknna if it even to the number of the batch
            if batch.shape[0] == batch_size:
                sparse_cknnas.append(cknna(batch, sparse_representation, topk=10))
            
            # Track neurons that are activated at least once
            if dead_neurons_count is None:
                dead_neurons_count = (representations != 0).sum(dim=0).cpu().long()
            else:
                dead_neurons_count += (representations != 0).sum(dim=0).cpu().long()

        # Aggregate metrics across all batches
        mae = torch.cat(mae, dim=0).cpu().numpy()
        cs = torch.cat(cs, dim=0).cpu().numpy()
        l0 = torch.cat(l0, dim=0).cpu().numpy()
        fvu = torch.cat(fvu, dim=0).cpu().numpy()
        cknnas = np.array(cknnas)
        sparse_mae = torch.cat(sparse_mae, dim=0).cpu().numpy()
        sparse_cs = torch.cat(sparse_cs, dim=0).cpu().numpy()
        sparse_l0 = torch.cat(sparse_l0, dim=0).cpu().numpy()
        sparse_fvu = torch.cat(sparse_fvu, dim=0).cpu().numpy()
        sparse_cknnas = np.array(sparse_cknnas)
        
        # Count neurons that were never activated
        number_of_dead_neurons = torch.where(dead_neurons_count == 0)[0].shape[0]

        # Log final metrics
        logger.info(f"Fraction of Variance Unexplained (FVU): {np.mean(fvu)} +/- {np.std(fvu)}")
        logger.info(f"Normalized MAE: {np.mean(mae)} +/- {np.std(mae)}")
        logger.info(f"Cosine similarity: {np.mean(cs)} +/- {np.std(cs)}")
        logger.info(f"L0 messure: {np.mean(l0)} +/- {np.std(l0)}")
        logger.info(f"CKNNA: {np.mean(cknnas)} +/- {np.std(cknnas)}")
        logger.info(f"Number of dead neurons: {number_of_dead_neurons}")
        logger.info(f"\nSparse Fraction of Variance Unexplained (FVU): {np.mean(sparse_fvu)} +/- {np.std(sparse_fvu)}")
        logger.info(f"Sparse Normalized MAE: {np.mean(sparse_mae)} +/- {np.std(sparse_mae)}")
        logger.info(f"Sparse Cosine similarity: {np.mean(sparse_cs)} +/- {np.std(sparse_cs)}")
        logger.info(f"Sparse L0 messure: {np.mean(sparse_l0)} +/- {np.std(sparse_l0)}")
        logger.info(f"Sparse CKNNA: {np.mean(sparse_cknnas)} +/- {np.std(sparse_cknnas)}")


def main(args):
    """
    Main function to load model and dataset, then extract and evaluate representations.
    
    Args:
        args (argparse.Namespace): Command line arguments.
    """
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Load the trained model
    model, mean_center, scaling_factor, target_norm = load_model(args.model)
    logger.info("Model loaded")
    
    # Load the dataset with appropriate preprocessing
    if ("text" in args.model and "text" in args.data) or ("image" in args.model and "image" in args.data):
        logger.info("Using model mean and scalling factor")    
        dataset = SAEDataset(args.data)
        dataset.mean = mean_center.cpu()
        dataset.scaling_factor = scaling_factor
    else:    
        logger.info("Computing mean and scalling factor")    
        dataset = SAEDataset(args.data, mean_center=True if mean_center.sum() != 0.0 else False, target_norm=target_norm)
        
    logger.info(f"Dataset loaded with length: {len(dataset)}")
    logger.info(f"Dataset mean center: {dataset.mean.mean()}, Scaling factor: {dataset.scaling_factor} with target norm {dataset.target_norm}")
    # Construct output filename from model and data names
    model_path_name = args.model.split("/")[-1].replace(".pt","")
    data_path_name = args.data.split("/")[-1].replace(".npy","")
    repr_file_name = os.path.join(args.output_path, f"{data_path_name}_{model_path_name}")
    
    # Extract representations and compute metrics
    get_representation(model, dataset, repr_file_name, args.batch_size)


if __name__ == "__main__":
    args = parse_args()
    main(args)