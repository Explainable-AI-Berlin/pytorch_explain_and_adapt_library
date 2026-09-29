import os
import json
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


def score_representations(model, dataset, batch_size) -> dict:
    """
    Score representation by progressivly extracting TopK actviations
    
    Args:
        model: The Sparse Autoencoder model to evaluate
        dataset: Dataset to process
        batch_size (int): Number of samples to process at once

    Returns:
        dict with results
        
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
    
    topk_list = [16]
    while topk_list[-1] <= model.n_latents:
        new_k = topk_list[-1] * 2
        if new_k > model.n_latents:
            break
        topk_list.append(new_k)
    if model.n_latents not in topk_list:
        topk_list.append(model.n_latents)
    
    logger.info(f"Calculating metrics for list of TopK: {topk_list}")
        
    
    model.eval()
    model.to(device)
    results = {}
    with torch.no_grad():
        # Create dataloader for batch processing
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                               shuffle=True, num_workers=0)
        
        # Iterate over the list
        for k in topk_list:
            # Lists to collect metrics for each batch
            l0 = []
            mae = []
            fvu = []
            cs = []
            cknnas = []
            dead_neurons_count = None
            
            # Process data in batches
            for idx, batch in enumerate(tqdm(dataloader, desc="Extracting representations")):
                batch = batch.to(device)
                
                # Forward pass through the model
                with torch.no_grad():
                    _, representations, info = model.encode(batch, topk_number=k)
                    assert (representations[0] != 0).sum() <= k, f"TopK {k} is not correct, got {representations[0].sum()} instead of {k}"
                    outputs = model.decode(representations, info)
                
                # Post-process outputs and batch
                batch = dataset.unprocess_data(batch.cpu()).to(device)
                outputs = dataset.unprocess_data(outputs.cpu()).to(device)
                
                # Calculate and collect metrics
                fvu.append(explained_variance_full(batch, outputs))
                mae.append(normalized_mean_absolute_error(batch, outputs))
                cs.append(torch.nn.functional.cosine_similarity(batch, outputs))
                l0.append(l0_messure(representations))
                # Only calculate the cknna if it even to the number of the batch
                if batch.shape[0] == batch_size:
                    cknnas.append(cknna(batch, representations, topk=10))
                
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
            
            # Count neurons that were never activated
            number_of_dead_neurons = torch.where(dead_neurons_count == 0)[0].shape[0]

            # Log final metrics
            logger.info(f"TopK: {k}")
            logger.info(f"Fraction of Variance Unexplained (FVU): {np.mean(fvu)} +/- {np.std(fvu)}")
            logger.info(f"Normalized MAE: {np.mean(mae)} +/- {np.std(mae)}")
            logger.info(f"Cosine similarity: {np.mean(cs)} +/- {np.std(cs)}")
            logger.info(f"L0 messure: {np.mean(l0)} +/- {np.std(l0)}")
            logger.info(f"CKNNA: {np.mean(cknnas)} +/- {np.std(cknnas)}")
            logger.info(f"Number of dead neurons: {number_of_dead_neurons}")
            
            # Store results in dictionary
            results[k] = {
                "fvu": (float(np.mean(fvu)), float(np.std(fvu))),
                "mae": (float(np.mean(mae)), float(np.std(mae))),
                "cs": (float(np.mean(cs)), float(np.std(cs))),
                "l0": (float(np.mean(l0)), float(np.std(l0))),
                "cknnas": (float(np.mean(cknnas)), float(np.std(cknnas))),
                "number_of_dead_neurons": number_of_dead_neurons
            }
    
    # Return results
    return results

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
    
    # # Score representations for each TopK
    scores = score_representations(model, dataset, args.batch_size)
    
    # Construct output filename from model and data names
    model_path_name = args.model.split("/")[-1].replace(".pt","")
    data_path_name = args.data.split("/")[-1].replace(".npy","")
    results_path = os.path.join(args.output_path, f"{data_path_name}_{model_path_name}.json")
    
    # Save results to JSON file
    with open(results_path, 'w') as f:
        json.dump(scores, f, indent=4)
    
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)