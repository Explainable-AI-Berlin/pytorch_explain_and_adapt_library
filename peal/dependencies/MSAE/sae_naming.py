import os
import json
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm

from sae import SAE
from utils import SAEDataset, set_seed

"""
Sparse Autoencoder Interpreter

This script computes similarity matrices between input vectors and decoder features
of a Sparse Autoencoder (SAE) model. These matrices can be used to interpret the
learned features and their relationship to the input data.
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
    Parse command line arguments for the similarity computation script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Compute similarity matrices for SAE model interpretation")
    parser.add_argument("-m", "--model", type=str, required=True, 
                       help="Path to the trained SAE model file (.pt)")
    parser.add_argument("-v", "--vocab", type=str, required=True, 
                       help="Path to the target data file (.npy)")
    parser.add_argument("-p", "--path-to-save", type=str, default=".", 
                       help="Directory path to save the similarity matrices")
    parser.add_argument("-b", "--batch-size", type=int, default=4096, 
                       help="Batch size for processing data")
    parser.add_argument("-s", "--seed", type=int, default=42, 
                       help="Random seed for reproducibility")
    parser.add_argument("-w", "--num-workers", type=int, default=12, 
                       help="Number of worker processes for data loading")
    parser.add_argument("--patch-diff", default=True,
                       help="Remove zero sae activation reconstruction vector")
    parser.add_argument("--verbose", action='store_true', help="Print found matches")
    return parser.parse_args()


def cosine_similarity_matrix(A: torch.Tensor, B: torch.Tensor, logit_scale: float=1.0) -> torch.Tensor:
    """
    Compute the cosine similarity matrix between two sets of vectors.
    
    This function normalizes both sets of vectors and computes their dot product,
    resulting in a matrix of cosine similarities between all pairs of vectors.
    
    Args:
        A (torch.Tensor): First set of vectors, shape [n, feature_dim]
        B (torch.Tensor): Second set of vectors, shape [m, feature_dim]
        logit_scale (float): Logit scale for CLIP models it is 100 but for default we set it to 1.0
    
    Returns:
        torch.Tensor: Cosine similarity matrix of shape [n, m], where each element [i,j]
                     represents the cosine similarity between vector A[i] and B[j]
    """
    # Normalize the vectors to unit length
    A_normalized = A / A.norm(dim=1, keepdim=True)
    B_normalized = B / B.norm(dim=1, keepdim=True)
    
    # Calculate cosine similarity matrix
    similarity = logit_scale * A_normalized @ B_normalized.t()
    return similarity


def compute_similarities(
    model: torch.nn.Module, 
    dataset: torch.utils.data.Dataset, 
    batch_size: int,
    patch_diff: bool = True,
    num_workers: int = 6
) -> np.ndarray:
    """
    Compute similarity matrices between dataset vectors and decoder features.
    
    This function:
    1. Processes the dataset in batches
    2. Computes two types of similarity matrices:
       - Standard: between input vectors and decoder features
       - Bias-adjusted: between input vectors and decoder features + bias
    3. Tracks agreement between top matches in both approaches
    
    Args:
        model (torch.nn.Module): The trained SAE model
        dataset (torch.utils.data.Dataset): Dataset containing input vectors
        batch_size (int): Batch size for processing
        num_workers (int): Number of data loading worker processes
        patch_diff (bool): In the `https://arxiv.org/abs/2407.14499` paper they didn't do it 
                            however we find it more interpretable (default: True)
        
    Returns:
        tuple: (standard_similarity_matrix, bias_similarity_matrix)
            - standard_similarity_matrix: Cosine similarities without bias, shape [n_inputs, n_features]
    """
    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Get decoder search space (the learned feature vectors)
    decoded_search_space = model.model.decoder.data + model.model.pre_bias
    decoded_search_space = model.postprocess(decoded_search_space)
    if patch_diff:
        zero_space = model.decode(torch.zeros(1,model.latent_dim, dtype=decoded_search_space.dtype, device=decoded_search_space.device))
        decoded_search_space -= zero_space
    logger.info(f"Input features: {len(dataset)}, Decoder features: {decoded_search_space.shape[0]}")
    
    # Initialize storage for similarity scores
    standard_similarities = []
     
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Computing similarities", unit="batch"):
            # In the paper we used `input_processed, _ = model.model.preprocess(batch_data.to(device))`
            # However due to not perfect reconstruction of the model, we simply use the CLIP text 
            # Compute similarity
            std_sim = cosine_similarity_matrix(batch_data, decoded_search_space)
            standard_similarities.append(std_sim.cpu().numpy())
            
    # Convert lists of batch results to complete matrices
    standard_matrix = np.concatenate(standard_similarities, axis=0).astype(np.float32)
    
    # Log results
    logger.info("Similarity computation complete")
    
    # Validate output shapes
    expected_shape = (len(dataset), decoded_search_space.shape[0])
    assert standard_matrix.shape == expected_shape, f"Standard similarity shape mismatch: {standard_matrix.shape} vs expected {expected_shape}"

    return standard_matrix


def main(args):
    """
    Main function to compute and save similarity matrices.
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Load the trained SAE model
    model = SAE(args.model)
    logger.info("Model loaded")
    
    # Load the vocabulary dataset
    dataset = SAEDataset(args.vocab, mean_center=False, target_norm=0.0)
    logger.info(f"Vocabulary dataset loaded with {len(dataset)}")
    
    # Compute both types of similarity matrices
    standard_matrix = compute_similarities(
        model, 
        dataset,
        patch_diff=args.patch_diff,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Prepare filenames for saving results
    data_path = os.path.join(
        args.path_to_save,
        f"Concept_Interpreter_{os.path.basename(args.model)}_{os.path.basename(args.vocab)}.npy".replace("/","~").replace(".pth","").replace(".pt","").replace(".npy","")
    )
    # Save the matrix
    np.save(data_path, standard_matrix)
    logger.info(f"Successfully saved similarity matrix to {data_path}")
    
    # Print per vocab best match
    if args.verbose:
        for vocab_id in range(standard_matrix.shape[0]):
            # Get the index of the most similar feature for each input vector
            best_match_index = np.argmax(standard_matrix[vocab_id])
            # Get the similarity score
            best_match_score = standard_matrix[vocab_id, best_match_index]
            # Print the result
            logger.info(f"Vocab ID: {vocab_id}/{standard_matrix.shape[0]}({float(vocab_id)/standard_matrix.shape[0]:.2f}), Best match SAE index: {best_match_index}, Score: {best_match_score}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
