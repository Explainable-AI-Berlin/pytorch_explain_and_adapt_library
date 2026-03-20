"""
Testing functions

Author: Luisa Gallée, Github: `https://github.com/XRad-Ulm/FunnyNodules`
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset_generator import create_attribute_dataloader


def test_contrastivity(model):
    import pandas as pd
    attributes = ["roundness", "spiculation", "edge_sharpness", "size_attr", "intensity", "internal_structure"]
    results = []
    for s in list(range(0, 100)):
        for attributes_idx, attr_name in enumerate(attributes):
            dl = create_attribute_dataloader(attr_name, base_seed=s)
            allimgs, allattributes, alltars = [], [], []
            for imgs, masks, attrs, targets, idxs in dl:
                for i in range(len(imgs)):
                    allimgs.append(imgs[i, 0])
                    allattributes.append(attrs[i])
                    alltars.append(targets[i])
            num_variations = 5 if attr_name != "internal_structure" else 2
            # fig, axs = plt.subplots(1,num_variations)
            # fig.suptitle(attr_name,fontsize=12)
            # for i in range(num_variations):
            #     axs[i].imshow(allimgs[i].cpu().numpy(), cmap="gray")
            #     axs[i].set_title("tar:"+str(alltars[i].item())+"a:"+str(allattributes[i][attributes_idx].item()),fontsize=12)
            #     axs[i].axis("off")
            # plt.show()
            # todo add test function of your model here
            _, test_acc_contra, test_attr_acc_contra = test(testmodel=model, data_loader=dl)
            entry = {"seed": s, "varied_attribute": attr_name, "test_acc_contra": float(test_acc_contra), }
            for i, attr in enumerate(attributes):
                entry[f"attr_acc_{attr}"] = float(test_attr_acc_contra[i].item())
            results.append(entry)
    results_df = pd.DataFrame(results)
    summary = results_df.groupby("varied_attribute")[
        ["test_acc_contra", "attr_acc_roundness", "attr_acc_spiculation", "attr_acc_edge_sharpness",
         "attr_acc_size_attr", "attr_acc_intensity", "attr_acc_internal_structure"]].agg(["mean", "std"])
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 0)
    pd.set_option('display.max_colwidth', None)
    print(summary.round(3))
    # === Plot model test accuracy by varied attribute ===
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # Left plot: test accuracy per varied attribute
    results_df.boxplot(column="test_acc_contra", by="varied_attribute", ax=axs[0], fontsize=12)
    axs[0].set_title("Model Test Accuracy by Varied Attribute", fontsize=12)
    axs[0].set_xlabel("Varied Attribute", fontsize=12)
    axs[0].set_ylabel("Accuracy", fontsize=12)
    # Right plot: heatmap-style barplot for attribute accuracies
    attr_cols = ["attr_acc_roundness", "attr_acc_spiculation", "attr_acc_edge_sharpness", "attr_acc_size_attr",
                 "attr_acc_intensity", "attr_acc_internal_structure"]
    # Compute mean accuracies for each varied_attribute
    attr_summary = results_df.groupby("varied_attribute")[attr_cols].mean()
    attr_summary.plot(kind="bar", ax=axs[1], fontsize=12)
    axs[1].set_title("Attribute Prediction Accuracies (mean per varied attribute)", fontsize=12)
    axs[1].set_xlabel("Varied Attribute", fontsize=12)
    axs[1].set_ylabel("Accuracy", fontsize=12)
    axs[1].legend(title="Predicted Attribute", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12)
    plt.tight_layout()
    plt.show()


def visualize_all_attribute_responses(model):
    attributes = ["roundness", "spiculation", "edge_sharpness", "size_attr", "intensity", "internal_structure"]
    seeds = list(range(0, 100))
    model.eval()
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    axs = axs.flatten()

    for i, attr in enumerate(attributes):
        all_varied = []
        all_true = []
        all_pred = []

        for s in seeds:
            dl = create_attribute_dataloader(attr, base_seed=s)
            varied_values = []
            true_targets = []
            pred_targets = []
            with torch.no_grad():
                for imgs, masks, attrs, targets, idxs in dl:
                    imgs = imgs.to("cuda", dtype=torch.float)
                    attrs = attrs.to("cuda", dtype=torch.float)
                    targets = targets.to("cuda", dtype=torch.float)
                    # todo add prediction of you model here
                    # e.g. model == "ResNetMT" with multitask [target, roundness, spiculation,...]
                    imgs = imgs.repeat(1, 3, 1, 1)
                    outputs = model(imgs)
                    pred = outputs[:, 0]

                    pred = pred.detach().cpu()
                    targets = targets.detach().cpu()

                    pred_targets.extend(pred.numpy().tolist())
                    true_targets.extend(targets.numpy().tolist())
                    varied_values.extend(attrs[:, attributes.index(attr)].cpu().numpy().tolist())
            order = np.argsort(varied_values)
            all_varied.append(np.array(varied_values)[order])
            all_true.append(np.array(true_targets)[order])
            all_pred.append(np.array(pred_targets)[order])

        all_varied = np.array([v.flatten() for v in all_varied])
        all_true = np.array([v.flatten() for v in all_true])
        all_pred = np.array([v.flatten() for v in all_pred])
        mean_true = all_true.mean(axis=0)
        mean_pred = all_pred.mean(axis=0)
        std_pred = all_pred.std(axis=0)
        x_vals = all_varied[0]
        print('visualize_all_attribute_responses')
        print(attr)
        print('x_vals ' + str(x_vals))
        print('mean_true ' + str(mean_true))
        print('mean_pred ' + str(mean_pred))
        ax = axs[i]
        ax.plot(x_vals, mean_true, 'o-', color='black', label='Ground Truth')
        ax.plot(x_vals, mean_pred, 's--', color='tab:blue', label='Predicted')
        ax.fill_between(x_vals, mean_pred - std_pred, mean_pred + std_pred, color='tab:blue', alpha=0.2)
        ax.set_title(attr, fontsize=12)
        ax.set_xlabel("Attribute Value (normalized 0–1)", fontsize=12)
        ax.set_ylabel("Target Value (normalized 0→1)", fontsize=12)
        ax.grid(True)
        if i == 0:
            ax.legend()

    plt.suptitle("Model Target Sensitivity", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def visualize_roundness_cs_interaction(model, img_size):
    model.eval()
    roundness_values = [1, 2, 3, 4, 5]
    seeds = list(range(0, 100))
    # store results separated by internal structure
    results = {0: {"true": [], "pred": []}, 1: {"true": [], "pred": []}}
    for s in seeds:
        dl = create_attribute_dataloader(attribute_name="roundness", base_seed=s, img_size=img_size, batch_size=1,
                                         fix_all_but_r_and_is=True)
        true_targets_cs = {0: [], 1: []}
        pred_targets_cs = {0: [], 1: []}
        for img, mask, attrs, target, _ in dl:
            cs_val = int(attrs[0, -1].item())
            img = img.to("cuda", dtype=torch.float)
            with torch.no_grad():
                # todo add prediction of you model here
                # e.g. model == "ResNetMT" with multitask [target, roundness, spiculation,...]
                img = img.repeat(1, 3, 1, 1)
                outputs = model(img)
                pred = outputs[:, 0]

                pred = pred.squeeze().cpu().item()
                pred_targets_cs[cs_val].append(pred)
                true_targets_cs[cs_val].append(target.item())
        for cs in [0, 1]:
            if len(true_targets_cs[cs]) > 0:
                results[cs]["true"].append(true_targets_cs[cs])
                results[cs]["pred"].append(pred_targets_cs[cs])
    plt.figure(figsize=(7, 5))
    for cs in [0, 1]:
        all_true = np.array(results[cs]["true"])
        all_pred = np.array(results[cs]["pred"])
        mean_true = all_true.mean(axis=0)
        mean_pred = all_pred.mean(axis=0)
        std_pred = all_pred.std(axis=0)
        color_gt = 'black' if cs == 0 else 'gray'
        color_pred = 'tab:blue' if cs == 0 else 'tab:orange'
        plt.plot(roundness_values, mean_true, 'o-', color=color_gt, label=f"GT is={cs}")
        plt.plot(roundness_values, mean_pred, 's--', color=color_pred, label=f"Pred is={cs}")
        plt.fill_between(roundness_values, mean_pred - std_pred, mean_pred + std_pred, color=color_pred, alpha=0.2)
        print('visualize_roundness_is_interaction')
        print('is ' + str(cs))
        print('roundness_values ' + str(roundness_values))
        print('mean_true ' + str(mean_true))
        print('mean_pred ' + str(mean_pred))

    plt.xlabel("Roundness")
    plt.ylabel("Target Value (normalized 0→1)")
    plt.title("Effect of Roundness on Target conditioned on IS with other attributes=3")
    plt.grid(True)
    plt.legend()
    plt.show()
