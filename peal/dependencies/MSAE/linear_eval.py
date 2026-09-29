import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

from sae import load_model
from utils import SAEDataset, set_seed, get_device

"""
Linear Probe Evaluation for Sparse Autoencoders

This script evaluates how well a Sparse Autoencoder preserves semantic information by:
1. Training a linear classifier (probe) on original data to predict class labels
2. Comparing predictions between original data and its reconstructions
3. Measuring discrepancy using KL divergence and argmax agreement

This approach helps quantify how much semantic/class information is preserved
in the autoencoder's latent space.
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
    Parse command line arguments for the linear probe evaluation.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Evaluate semantic preservation in SAE reconstructions using a linear probe")
    parser.add_argument("-m", "--model", type=str, required=True, 
                       help="Path to the trained SAE model file")
    parser.add_argument("-d", "--data", type=str, required=True, 
                       help="Path to training data embeddings file (.npy)")
    parser.add_argument("-t", "--target", type=str, required=True, 
                       help="Path to training data labels file (.txt)")
    parser.add_argument("-e", "--eval_data", type=str, required=True, 
                       help="Path to evaluation data embeddings file (.npy)")
    parser.add_argument("-o", "--eval_target", type=str, required=True, 
                       help="Path to evaluation data labels file (.txt)")
    parser.add_argument("-b", "--batch-size", type=int, default=512, 
                       help="Batch size for training and evaluation")
    parser.add_argument("-s", "--seed", type=int, default=42, 
                       help="Random seed for reproducibility")
    parser.add_argument("-w", "--num-workers", type=int, default=8, 
                       help="Number of worker processes for data loading")

    return parser.parse_args()


def valid_linear_probe(model: torch.nn.Module, linear_probe: torch.nn.Module, 
                      dataset: torch.utils.data.Dataset, device: torch.device, 
                      batch_size: int) -> tuple[list[float], list[float]]:
    """
    Evaluate how well reconstructed data preserves the semantic information in original data.
    
    For each batch of samples:
    1. Gets model reconstructions
    2. Passes both original and reconstructed data through the linear probe
    3. Compares predictions using KL divergence and argmax agreement
    
    Args:
        model (torch.nn.Module): The trained autoencoder model
        linear_probe (torch.nn.Module): The trained linear classifier
        dataset (torch.utils.data.Dataset): Dataset containing the data to evaluate
        device (torch.device): Device to run the evaluation on
        batch_size (int): Batch size for processing
        
    Returns:
        tuple: (kl_divergences, argmax_agreements)
            - kl_divergences: List of KL divergence values between prediction distributions
            - argmax_agreements: List of binary values (1=agree, 0=disagree) for prediction classes
    """
    linear_probe.eval()
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2,
        pin_memory=True
    )

    with torch.no_grad():
        kl = []
        arg_max = []

        for batch in tqdm(dataloader, desc="Evaluating semantic preservation"):
            # Move batch to device
            data = batch.to(device)
            
            # Get reconstruction from model
            _, _, data_reconstruction, _ = model(data)

            # Normalize data (cosine normalization)
            data_norm = data / data.norm(dim=-1, keepdim=True)
            data_reconstruction_norm = data_reconstruction / data_reconstruction.norm(dim=-1, keepdim=True)

            # Get predictions from linear probe
            data_predicted = linear_probe(data_norm)
            data_reconstruction_predicted = linear_probe(data_reconstruction_norm)

            # Apply softmax to get proper probability distributions
            data_predicted_prob = torch.nn.functional.softmax(data_predicted, dim=1)
            data_reconstruction_predicted_prob = torch.nn.functional.softmax(data_reconstruction_predicted, dim=1)

            # Calculate KL divergence for each sample in the batch
            for i in range(data.shape[0]):
                # Extract individual sample predictions
                orig_pred = data_predicted_prob[i:i+1]
                recon_pred = data_reconstruction_predicted_prob[i:i+1]
                
                # KL divergence between the probability distributions
                kl_value = torch.nn.functional.kl_div(
                    recon_pred.log(),
                    orig_pred,
                    reduction='batchmean'
                ).item()
                kl.append(kl_value)
                
                # Check if the predicted class is the same (argmax agreement)
                orig_class = torch.argmax(data_predicted[i])
                recon_class = torch.argmax(data_reconstruction_predicted[i])
                arg_max.append((orig_class == recon_class).item())

    return kl, arg_max


def evaluate_linear_probe(linear_probe: torch.nn.Module, 
                         dataset: torch.utils.data.Dataset, 
                         device: torch.device) -> float:
    """
    Evaluate the linear probe's performance on a dataset using macro F1 score.
    
    Args:
        linear_probe (torch.nn.Module): The trained linear classifier
        dataset (torch.utils.data.Dataset): Dataset containing data and targets
        device (torch.device): Device to run the evaluation on
        
    Returns:
        float: Macro-averaged F1 score across all classes
    """
    linear_probe.eval()
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=2
    )

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for data, target in tqdm(dataloader, total=len(dataloader), desc="Evaluating classifier"):
            data = data.to(device)
            target = target.to(device)

            # Normalize input data
            data = data / data.norm(dim=-1, keepdim=True)
            
            # Get predictions
            outputs = linear_probe(data)
            _, predicted = torch.max(outputs, 1)
            
            # Collect targets and predictions
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate F1 score once on all predictions
    return f1_score(all_targets, all_predictions, average="macro")


def train_linear_probe(model: torch.nn.Module, dataset: torch.utils.data.Dataset, 
                      eval_dataset: torch.utils.data.Dataset, target_path: str, 
                      eval_target_path: str, device: torch.device, 
                      batch_size: int, num_workers: int) -> torch.nn.Module:
    """
    Train a linear classifier (probe) on data embeddings to predict class labels.
    
    This function:
    1. Prepares datasets with class labels
    2. Trains a linear classifier using cross-entropy loss
    3. Implements early stopping based on evaluation F1 score
    4. Returns the best model based on validation performance
    
    Args:
        model (torch.nn.Module): The trained autoencoder model (used for dimensions)
        dataset (torch.utils.data.Dataset): Dataset containing training data
        eval_dataset (torch.utils.data.Dataset): Dataset containing evaluation data
        target_path (str): Path to file containing training data labels
        eval_target_path (str): Path to file containing evaluation data labels
        device (torch.device): Device to run the training on
        batch_size (int): Batch size for training
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        torch.nn.Module: The trained linear probe (classifier)
    """
    # Create data loaders to efficiently load data in batches
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    # Load and prepare training data
    logger.info("Loading and preparing data...")
    data_list = []
    for batch in tqdm(train_dataloader, desc="Loading training data"):
        data_list.append(batch)
    data = torch.cat(data_list, dim=0)
    
    # Load and prepare evaluation data
    eval_data_list = []
    for batch in tqdm(eval_dataloader, desc="Loading evaluation data"):
        eval_data_list.append(batch)
    eval_data = torch.cat(eval_data_list, dim=0)

    # Load and prepare training targets
    with open(target_path, "r") as f:
        target_dataset = f.readlines()
    target_dataset = [x.strip() for x in target_dataset]

    # Create mapping from text labels to numeric indices
    unique_texts = list(dict.fromkeys(target_dataset))
    unique_target = {text: idx for idx, text in enumerate(unique_texts)}
    logger.info(f"Number of unique labels: {len(unique_target)}")
    
    # Convert text labels to numeric indices
    target = torch.empty(len(target_dataset), dtype=torch.long)
    for idx, label in enumerate(target_dataset):
        target[idx] = unique_target[label]

    # Verify data and target dimensions match
    assert data.shape[0] == len(target), f"Data shape {data.shape} and target shape {target.shape} do not match"

    # Load and prepare evaluation targets
    with open(eval_target_path, "r") as f:
        eval_target_dataset = f.readlines()
    eval_target_dataset = [x.strip() for x in eval_target_dataset]

    # Verify evaluation target labels match training target labels
    assert set(list(dict.fromkeys(eval_target_dataset))) <= set(unique_texts), "Evaluation target labels not found in training target labels"
    
    # Convert evaluation text labels to numeric indices
    eval_target = torch.empty(len(eval_target_dataset), dtype=torch.long)
    for idx, label in enumerate(eval_target_dataset):
        eval_target[idx] = unique_target[label]

    # Create datasets and dataloaders for training
    train_tensor_dataset = torch.utils.data.TensorDataset(data, target)
    train_dataloader = torch.utils.data.DataLoader(
        train_tensor_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    eval_tensor_dataset = torch.utils.data.TensorDataset(eval_data, eval_target)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_tensor_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Define training hyperparameters
    epochs = 100
    patience = 5  # Number of epochs to wait for improvement
    n_classes = len(unique_target)
    
    # Create linear probe model
    linear_probe = torch.nn.Sequential(
        torch.nn.Linear(model.n_inputs, n_classes),
    )
    linear_probe.to(device)

    # Define optimizer, scheduler and loss function
    optimizer = torch.optim.AdamW(linear_probe.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    criterion = torch.nn.CrossEntropyLoss()

    logger.info(f"Training linear probe with {n_classes} classes for {epochs} epochs")
    best_acc = 0.0
    best_model = None
    patience_counter = 0

    # Training loop
    for epoch in range(epochs):
        # Training phase
        linear_probe.train()
        train_loss = 0.0
        
        for embeddings, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            # Normalize embeddings
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            
            # Forward pass
            outputs = linear_probe(embeddings)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        # Evaluation phase
        acc = evaluate_linear_probe(linear_probe, eval_tensor_dataset, device)
        if scheduler is not None:
            scheduler.step(acc)

        logger.info(f"Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Eval F1: {acc:.4f}")

        # Save best model and implement early stopping
        if acc > best_acc:
            best_acc = acc
            best_model = linear_probe.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered. Best F1: {best_acc:.4f}")
                break

    # Load best model
    linear_probe.load_state_dict(best_model)
    logger.info(f"Linear probe training complete with best eval F1: {best_acc:.4f}")

    return linear_probe


def main(args):
    """
    Main function to run the linear probe evaluation pipeline.
    
    This function:
    1. Loads the trained SAE model
    2. Prepares the datasets
    3. Trains a linear probe classifier
    4. Evaluates semantic preservation between original and reconstructed data
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Get the device to use for training
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load the trained SAE model
    model, mean_center, scaling_factor, target_norm = load_model(args.model)
    model.to(device)
    model.eval()
    logger.info(f"Model loaded: {args.model}")
    
    # Load the dataset with appropriate preprocessing
    if ("text" in args.model and "text" in args.data) or ("image" in args.model and "image" in args.data):
        logger.info("Using model mean and scalling factor")    
        train_ds = SAEDataset(args.data)
        train_ds.mean = mean_center.cpu()
        train_ds.scaling_factor = scaling_factor
    else:    
        logger.info("Computing mean and scalling factor")    
        train_ds = SAEDataset(args.data, mean_center=True if mean_center.sum() != 0.0 else False, target_norm=target_norm)
    logger.info(f"Training dataset loaded with {len(train_ds)} samples")
    
    if ("text" in args.model and "text" in args.eval_data) or ("image" in args.model and "image" in args.eval_data):
        logger.info("Using model mean and scalling factor")    
        eval_ds = SAEDataset(args.eval_data)
        eval_ds.mean = mean_center.cpu()
        eval_ds.scaling_factor = scaling_factor
    else:    
        logger.info("Computing mean and scalling factor")    
        eval_ds = SAEDataset(args.eval_data, mean_center=True if mean_center.sum() != 0.0 else False, target_norm=target_norm)
    logger.info(f"Evaluation dataset loaded with {len(eval_ds)} samples")

    # Train the linear probe classifier
    logger.info("Training linear probe classifier...")
    linear_probe = train_linear_probe(
        model, train_ds, eval_ds, args.target, args.eval_target, 
        device, args.batch_size, args.num_workers
    )
    
    # Evaluate semantic preservation
    logger.info("Evaluating semantic preservation in reconstructions...")
    train_kl, train_arg_max = valid_linear_probe(model, linear_probe, train_ds, device, args.batch_size)
    eval_kl, eval_arg_max = valid_linear_probe(model, linear_probe, eval_ds, device, args.batch_size)

    # Report results
    logger.info("Results Summary:")
    logger.info("Training Set:")
    logger.info(f"  KL Divergence: {np.mean(train_kl):.4f} ± {np.std(train_kl):.4f}")
    logger.info(f"  Class Prediction Agreement: {np.mean(train_arg_max)*100:.2f}% ± {np.std(train_arg_max)*100:.2f}%")
    
    logger.info("Evaluation Set:")
    logger.info(f"  KL Divergence: {np.mean(eval_kl):.4f} ± {np.std(eval_kl):.4f}")
    logger.info(f"  Class Prediction Agreement: {np.mean(eval_arg_max)*100:.2f}% ± {np.std(eval_arg_max)*100:.2f}%")
    
    logger.info("Interpretation:")
    logger.info("  - Lower KL Divergence indicates better preservation of semantic information")
    logger.info("  - Higher Class Prediction Agreement indicates better preservation of class identity")


if __name__ == "__main__":
    args = parse_args()
    main(args)