import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

from sae import load_model
from utils import set_seed

"""
SAE Alignment Analysis Tool from paper:
`Sparse Autoencoders Trained on the Same Data Learn Different Features`
https://anonymous.4open.science/r/sae_overlap-51F1/hungarian_aligmnent.py

This script computes the alignment between two trained Sparse Autoencoders (SAEs)
using the Hungarian algorithm. It measures how similar the learned features are
by computing the optimal matching between neurons across the two models.

The alignment score indicates how well the feature representations align between
the models, with higher scores indicating better alignment.
"""

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
    parser = argparse.ArgumentParser(description="Analyze alignment between two SAE models")
    parser.add_argument("-f", "--sae_1", type=str, required=True, 
                       help="Path to the first SAE model file (.pt)")
    parser.add_argument("-s", "--sae_2", type=str, required=True, 
                       help="Path to the second SAE model file (.pt)")
    parser.add_argument("-b", "--batch-size", type=int, default=512, 
                       help="Batch size for cost matrix computation")
    parser.add_argument("-r", "--seed", type=int, default=42, 
                       help="Random seed for reproducibility")
    parser.add_argument("--decoder", action="store_true",
                       help="Use decoder weights instead of encoder weights")
    
    return parser.parse_args()

def matching(sae_1_weight: torch.Tensor, sae_2_weight: torch.Tensor, 
             batch_size: int) -> tuple[float, tuple[np.ndarray, np.ndarray]]:
    """
    Compute optimal matching between neurons of two SAEs using the Hungarian algorithm.
    
    This function computes a cost matrix of dot products between normalized weight vectors
    from both models, then applies the linear sum assignment algorithm (Hungarian method)
    to find the optimal one-to-one matching that maximizes the overall similarity.
    
    Args:
        sae_1_weight (torch.Tensor): Normalized weight matrix from first SAE
                                    Shape: [n_neurons_1, feature_dim]
        sae_2_weight (torch.Tensor): Normalized weight matrix from second SAE
                                    Shape: [n_neurons_2, feature_dim]
        batch_size (int): Batch size for processing large matrices
        
    Returns:
        tuple:
            - float: Average similarity score for the optimal matching
            - tuple[np.ndarray, np.ndarray]: Indices of matched neurons (row_indices, col_indices)
    """
    # Calculate number of batches needed
    n_batches = (sae_1_weight.shape[0] + batch_size - 1) // batch_size
    logger.info(f"Number of batches: {n_batches}")
    
    # Initialize cost matrix
    cost_matrix_skips = torch.zeros(sae_1_weight.shape[0], sae_2_weight.shape[0], 
                                    device="cpu", requires_grad=False)
    
    # Compute cost matrix in batches to avoid memory issues
    for i in tqdm(range(n_batches), desc="Calculating cost matrix"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, sae_1_weight.shape[0])
        # Compute dot products (cosine similarity for normalized vectors)
        value = sae_1_weight[start_idx:end_idx] @ sae_2_weight.T
        cost_matrix_skips[start_idx:end_idx] = value.cpu()
    
    # Handle any NaN values
    cost_matrix_skips = torch.nan_to_num(cost_matrix_skips, nan=0)
    
    # Apply Hungarian algorithm to find optimal matching
    # Note: We set maximize=True because we want to maximize similarity
    row_ind_skips, col_ind_skips = linear_sum_assignment(
        cost_matrix_skips.detach().numpy(), maximize=True)
    
    # Calculate average similarity for the optimal matching
    score = cost_matrix_skips[row_ind_skips, col_ind_skips].mean().item()
    index = (row_ind_skips, col_ind_skips)
    
    return score, index

def main(args):
    """
    Main function for analyzing alignment between two SAE models.
    
    This function:
    1. Loads two pre-trained SAE models
    2. Normalizes either decoder or encoder weights based on user preference
    3. Computes the optimal matching between neurons and the alignment score
    4. Logs the results
    
    Args:
        args (argparse.Namespace): Command line arguments from parse_args()
    """
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Load the trained models
    model_1 = load_model(args.sae_1)[0]
    model_1.eval()
    logger.info(f"Loaded first model")
    
    model_2 = load_model(args.sae_2)[0]
    model_2.eval()
    logger.info(f"Loaded second model")
    
    # Determine whether to use decoder or encoder weights
    logger.info("Using decoder weights" if args.decoder else "Using encoder weights")
    
    if args.decoder:
        # Normalize decoder weights to unit norm
        model_1_weight = model_1.decoder.data / model_1.decoder.data.norm(dim=1, keepdim=True)
        model_2_weight = model_2.decoder.data / model_2.decoder.data.norm(dim=1, keepdim=True)
    else:
        # Normalize encoder weights to unit norm
        # In the original implementation encoder columns were normalized where it should be rows
        model_1_weight = model_1.encoder.data / model_1.encoder.data.norm(dim=0, keepdim=True)
        model_2_weight = model_2.encoder.data / model_2.encoder.data.norm(dim=0, keepdim=True)
        model_1_weight = model_1_weight.T
        model_2_weight = model_2_weight.T
    
    # Compute the matching and alignment score
    score, _ = matching(model_1_weight, model_2_weight, args.batch_size)
    
    # Log the results
    logger.info(f"Hungarian Alignment Score: {score:.4f}")
    logger.info("Score interpretation: Higher values (closer to 1.0) indicate better alignment between models")


if __name__ == "__main__":
    args = parse_args()
    main(args)