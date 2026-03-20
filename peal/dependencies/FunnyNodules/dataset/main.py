"""
Main function

Author: Luisa Gallée, Github: `https://github.com/XRad-Ulm/FunnyNodules`
"""

import random

from dataset_generator import generate_nodule, create_dataloader, calculate_target, visualize_nodule

if __name__ == "__main__":
    # Create dataloader
    dataloader = create_dataloader(num_samples=1800, batch_size=16, img_size=224)

    # show some samples with fixed seed
    samples = []
    meta = []
    idx = 0
    for seed in range(100):
        random.seed(seed)
        internal_structure = random.choice([0, 1])
        edge_sharpness = random.choice([1, 2, 3, 4, 5])
        intensity = random.choice([1, 2, 3, 4, 5])
        roundness = random.choice([1, 2, 3, 4, 5])
        size_attr = random.choice([1, 2, 3, 4, 5])
        spiculation = random.choice([1, 2, 3, 4, 5])
        idx += 1
        img, mask, attribute_ROIs = generate_nodule(128, roundness, spiculation, edge_sharpness, size_attr, intensity,
                                                    internal_structure, seed=seed)
        target = calculate_target(roundness, spiculation, edge_sharpness, size_attr, intensity, internal_structure)
        meta.append({"roundness": roundness, "spiculation": spiculation, "edge_sharpness": edge_sharpness,
                     "size_attr": size_attr, "intensity": intensity, "internal_structure": internal_structure,
                     "target": target})
        samples.append((img, mask, attribute_ROIs, meta[-1]))
    for i, (img, mask, attribute_ROIs, labels) in enumerate(samples):
        visualize_nodule(img, mask, attribute_ROIs, labels)
