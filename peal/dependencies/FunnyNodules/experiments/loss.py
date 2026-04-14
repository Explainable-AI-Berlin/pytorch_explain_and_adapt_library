"""
Loss function combining attribute, target and prototype learning.

Author: Luisa Gallée, Github: `https://github.com/XRad-Ulm/HierViT`
"""
import sys

import matplotlib.pyplot as plt
import torch
from torch import nn


def model_loss(y, y_pred, x, x_recon, lam_recon, attr_gt, attr_pred, lam_attr, epoch,
               dists_to_protos, sample_id, idx_with_attri, max_dist, args):
    if args.dataset in ["LIDC","FunnyNodules"]:
        if args.threeD:
            L_recon = nn.MSELoss()(x_recon.permute(0,1,4,2,3), x)
        else:
            L_recon = nn.MSELoss()(torch.squeeze(x_recon), torch.squeeze(x))

    elif args.dataset in ["Chexbert", "derm7pt"]:
        L_recon = 0


    if args.dataset == "FunnyNodules":
        L_pred = nn.MSELoss()(y_pred, y)
    elif args.dataset == "LIDC":
        if args.base_model == "ViT":
            y = torch.argmax(y, axis=1).to(torch.float32)
            y /= 4.0
            L_pred = nn.MSELoss()(torch.squeeze(y_pred), y)
        else:
            L_pred = nn.KLDivLoss(reduction="batchmean")(y_pred, y)
    elif args.dataset == "derm7pt":
        weights = [1.0/0.568743818,1.0/0.044510386,1.0/0.095944609,1.0/0.041543027,1.0/0.24925816]
        class_weights = torch.FloatTensor(weights).cuda()
        L_pred = nn.CrossEntropyLoss(weight=class_weights)(y_pred,y)
    elif args.dataset == "Chexbert":
        L_pred = nn.BCELoss()(torch.squeeze(y_pred),y)

    batchidx_with_attri = []
    for i in range(len(sample_id)):
        if sample_id[i] in idx_with_attri:
            batchidx_with_attri.append(i)
    L_attr = 0.0
    if args.dataset == "derm7pt":
        weights = [[1.0/0.395647873,1.0/0.376854599,1.0/0.227497527],
                   [1.0/0.807121662,1.0/0.192878338],
                   [1.0/0.8140455,1.0/0.115727003,1.0/0.070227498],
                   [1.0/0.581602374,1.0/0.116716123,1.0/0.301681503],
                   [1.0/0.645895153,1.0/0.105835806,1.0/0.248269041],
                   [1.0/0.226508408,1.0/0.330365974,1.0/0.443125618],
                   [1.0/0.74975272,1.0/0.25024728]]
        for i in range(7):
            class_weights = torch.FloatTensor(weights[i]).cuda()
            L_attr += nn.CrossEntropyLoss(weight=class_weights)(attr_pred[i][batchidx_with_attri], attr_gt[batchidx_with_attri][:, i])
    else:
        for i in range(attr_gt.shape[-1]):
            if args.dataset == "LIDC":
                L_attr += nn.MSELoss()(attr_pred[batchidx_with_attri, i], attr_gt[batchidx_with_attri, i])
            elif args.dataset == "Chexbert":
                L_attr += nn.BCELoss()(attr_pred[batchidx_with_attri, i], attr_gt[batchidx_with_attri, i])
            elif args.dataset == "FunnyNodules":
                L_attr += nn.MSELoss()(attr_pred[batchidx_with_attri, i], attr_gt[batchidx_with_attri, i])

    L_sep = 0.0
    L_cluster_allmean = 0.0
    L_cluster_allcpsi = 0.0


    if epoch < args.warmup:
        if args.onlyTar:
            total_loss = L_pred + lam_recon * L_recon
        else:
            total_loss = L_pred + lam_recon * L_recon + lam_attr * L_attr
    else:
        if len(batchidx_with_attri) > 0:
            for capsule_idx in range(len(dists_to_protos)):
                if args.dataset == "derm7pt":
                    if not capsule_idx in [1,6]:
                        idx0 = torch.squeeze((attr_gt[batchidx_with_attri, capsule_idx] == 0).nonzero(),dim=-1).cpu().detach().numpy()
                        idx1 = torch.squeeze((attr_gt[batchidx_with_attri, capsule_idx] == 1).nonzero(),dim=-1).cpu().detach().numpy()
                        idx2 = torch.squeeze((attr_gt[batchidx_with_attri, capsule_idx] == 2).nonzero(),dim=-1).cpu().detach().numpy()
                        idxs = [idx0, idx1, idx2]
                    else:
                        idx0 = torch.squeeze((attr_gt[batchidx_with_attri, capsule_idx] == 0).nonzero(),dim=-1).cpu().detach().numpy()
                        idx1 = torch.squeeze((attr_gt[batchidx_with_attri, capsule_idx] == 1).nonzero(),dim=-1).cpu().detach().numpy()
                        idxs = [idx0, idx1]
                    L_cluster = 0
                    L_sep_loss = 0
                    for idxi in range(len(idxs)):
                        if len(idxs[idxi]) > 0:
                            L_cluster += torch.mean(
                                torch.squeeze(dists_to_protos[capsule_idx][batchidx_with_attri][idxs[idxi], idxi]))
                    L_cluster_allcpsi += (L_cluster / len(batchidx_with_attri))
                if args.dataset == "FunnyNodules":
                    if capsule_idx < 5:
                        idx0 = (attr_gt[batchidx_with_attri, capsule_idx] < 0.125).nonzero()
                        idx1 = ((attr_gt[batchidx_with_attri, capsule_idx] >= 0.125) & (
                                attr_gt[batchidx_with_attri, capsule_idx] < 0.375)).nonzero()
                        idx2 = ((attr_gt[batchidx_with_attri, capsule_idx] >= 0.375) & (
                                attr_gt[batchidx_with_attri, capsule_idx] < 0.625)).nonzero()
                        idx3 = ((attr_gt[batchidx_with_attri, capsule_idx] >= 0.625) & (
                                attr_gt[batchidx_with_attri, capsule_idx] < 0.875)).nonzero()
                        idx4 = (attr_gt[batchidx_with_attri, capsule_idx] >= 0.875).nonzero()
                        idxs = [idx0, idx1, idx2, idx3, idx4]
                        L_cluster = 0

                        for idxi in range(len(idxs)):
                            if len(idxs[idxi]) > 0:
                                L_cluster += torch.mean(
                                    torch.squeeze(dists_to_protos[capsule_idx][batchidx_with_attri][idxs[idxi], idxi]))

                        L_cluster_allcpsi += (L_cluster / len(batchidx_with_attri))

                        # L_sep_loss = sep_loss(max_dist=max_dist,
                        #                       selected_dists=dists_to_protos[capsule_idx][batchidx_with_attri],
                        #                       indices=[idx0, idx1, idx2, idx3, idx4])

                        # L_sep += L_sep_loss / (len(batchidx_with_attri) * 4)
                    else:
                        idx0 = (attr_gt[batchidx_with_attri, capsule_idx] < 0.5).nonzero()
                        idx1 = (attr_gt[batchidx_with_attri, capsule_idx] > 0.5).nonzero()
                        idxs = [idx0, idx1]
                        L_cluster = 0

                        for idxi in range(len(idxs)):
                            if len(idxs[idxi]) > 0:
                                L_cluster += torch.mean(
                                    torch.squeeze(dists_to_protos[capsule_idx][batchidx_with_attri][idxs[idxi], idxi]))

                        L_cluster_allcpsi += (L_cluster / len(batchidx_with_attri))

                        # L_sep_loss = sep_loss(max_dist=max_dist,
                        #                       selected_dists=dists_to_protos[capsule_idx][batchidx_with_attri],
                        #                       indices=[idx0, idx1, idx2, idx3, idx4])

                        # L_sep += L_sep_loss / (len(batchidx_with_attri) * 1)
                if args.dataset == "LIDC":
                    if capsule_idx in [0, 3, 4, 5, 6, 7]:
                        idx0 = (attr_gt[batchidx_with_attri, capsule_idx] < 0.125).nonzero()
                        idx1 = ((attr_gt[batchidx_with_attri, capsule_idx] >= 0.125) & (
                                attr_gt[batchidx_with_attri, capsule_idx] < 0.375)).nonzero()
                        idx2 = ((attr_gt[batchidx_with_attri, capsule_idx] >= 0.375) & (
                                attr_gt[batchidx_with_attri, capsule_idx] < 0.625)).nonzero()
                        idx3 = ((attr_gt[batchidx_with_attri, capsule_idx] >= 0.625) & (
                                attr_gt[batchidx_with_attri, capsule_idx] < 0.875)).nonzero()
                        idx4 = (attr_gt[batchidx_with_attri, capsule_idx] >= 0.875).nonzero()
                        idxs = [idx0, idx1, idx2, idx3, idx4]
                        L_cluster = 0

                        for idxi in range(len(idxs)):
                            if len(idxs[idxi]) > 0:
                                L_cluster += torch.mean(
                                    torch.squeeze(dists_to_protos[capsule_idx][batchidx_with_attri][idxs[idxi], idxi]))

                        L_cluster_allcpsi += (L_cluster / len(batchidx_with_attri))

                        L_sep_loss = sep_loss(max_dist=max_dist, selected_dists=dists_to_protos[capsule_idx][batchidx_with_attri],indices=[idx0,idx1,idx2,idx3,idx4])

                        L_sep += L_sep_loss / (len(batchidx_with_attri) * 4)

                    elif capsule_idx == 1:
                        idx0 = (attr_gt[batchidx_with_attri, capsule_idx] < 0.16).nonzero()
                        idx1 = ((attr_gt[batchidx_with_attri, capsule_idx] >= 0.16) & (
                                attr_gt[batchidx_with_attri, capsule_idx] < 0.49)).nonzero()
                        idx2 = ((attr_gt[batchidx_with_attri, capsule_idx] >= 0.49) & (
                                attr_gt[batchidx_with_attri, capsule_idx] < 0.82)).nonzero()
                        idx3 = (attr_gt[batchidx_with_attri, capsule_idx] >= 0.82).nonzero()
                        idxs = [idx0, idx1, idx2, idx3]
                        L_cluster = 0

                        for idxi in range(len(idxs)):
                            if len(idxs[idxi]) > 0:
                                L_cluster += torch.mean(
                                    torch.squeeze(dists_to_protos[capsule_idx][batchidx_with_attri][idxs[idxi], idxi]))

                        L_cluster_allcpsi += (L_cluster / len(batchidx_with_attri))

                        L_sep_loss = sep_loss(max_dist=max_dist,
                                              selected_dists=dists_to_protos[capsule_idx][batchidx_with_attri],
                                              indices=[idx0, idx1, idx2, idx3])

                        L_sep += L_sep_loss / (len(batchidx_with_attri) * 3)

                    elif capsule_idx == 2:
                        idx0 = (attr_gt[batchidx_with_attri, capsule_idx] < 0.1).nonzero()
                        idx1 = ((attr_gt[batchidx_with_attri, capsule_idx] >= 0.1) & (
                                attr_gt[batchidx_with_attri, capsule_idx] < 0.3)).nonzero()
                        idx2 = ((attr_gt[batchidx_with_attri, capsule_idx] >= 0.3) & (
                                attr_gt[batchidx_with_attri, capsule_idx] < 0.5)).nonzero()
                        idx3 = ((attr_gt[batchidx_with_attri, capsule_idx] >= 0.5) & (
                                attr_gt[batchidx_with_attri, capsule_idx] < 0.7)).nonzero()
                        idx4 = ((attr_gt[batchidx_with_attri, capsule_idx] >= 0.7) & (
                                attr_gt[batchidx_with_attri, capsule_idx] < 0.9)).nonzero()
                        idx5 = (attr_gt[batchidx_with_attri, capsule_idx] >= 0.9).nonzero()
                        idxs = [idx0, idx1, idx2, idx3, idx4, idx5]
                        L_cluster = 0

                        for idxi in range(len(idxs)):
                            if len(idxs[idxi]) > 0:
                                L_cluster += torch.mean(
                                    torch.squeeze(dists_to_protos[capsule_idx][batchidx_with_attri][idxs[idxi], idxi]))

                        L_cluster_allcpsi += (L_cluster / len(batchidx_with_attri))

                        L_sep_loss = sep_loss(max_dist=max_dist,
                                              selected_dists=dists_to_protos[capsule_idx][batchidx_with_attri],
                                              indices=[idx0, idx1, idx2, idx3, idx4, idx5])

                        L_sep += L_sep_loss / (len(batchidx_with_attri) * 5)
                elif args.dataset == "Chexbert":
                    idx0 = (attr_gt[batchidx_with_attri, capsule_idx] == 0).nonzero()
                    idx1 = (attr_gt[batchidx_with_attri, capsule_idx] == 1).nonzero()
                    idxs = [idx0, idx1]
                    L_cluster = 0
                    for idxi in range(len(idxs)):
                        if len(idxs[idxi]) > 0:
                            L_cluster += torch.sum(
                                torch.squeeze(dists_to_protos[capsule_idx][batchidx_with_attri][idxs[idxi], idxi])[
                                    0])

                    L_cluster_allcpsi += (L_cluster / len(batchidx_with_attri))

                    L_sep_loss = sep_loss(max_dist=max_dist,
                                          selected_dists=dists_to_protos[capsule_idx][batchidx_with_attri],
                                          indices=[idx0, idx1])

                    L_sep += L_sep_loss / (len(batchidx_with_attri) * 1)

        L_sep /= len(dists_to_protos)
        L_cluster_allmean = L_cluster_allcpsi / len(dists_to_protos)
        if args.dataset == "LIDC":
            lam = 0.01#0.001#0.01# orig 0.01
            if args.base_model == "ViT":
                total_loss = L_pred + lam_recon * L_recon + lam_attr * L_attr + lam * (1 / 8) * (
                        L_cluster_allmean)
            else:
                total_loss = L_pred + lam_recon * L_recon + lam_attr * L_attr + lam * (1 / 8) * (
                    L_cluster_allmean + 0.1 * L_sep)
        elif args.dataset == "FunnyNodules":
            lam = 0.01
            if args.base_model == "ViT":
                total_loss = L_pred + lam_recon * L_recon + lam_attr * L_attr + lam * (1 / 6) * (
                    L_cluster_allmean)
            else:
                total_loss = L_pred + lam_recon * L_recon + lam_attr * L_attr + lam * (1 / 6) * (
                    L_cluster_allmean + 0.1 * L_sep)
        elif args.dataset == "derm7pt":
            lam = 0.01
            total_loss = L_pred + lam_recon * L_recon + lam_attr * L_attr + lam * (1 / 7) * (
                    L_cluster_allmean)
        elif args.dataset == "Chexbert":
            total_loss = L_pred + lam_recon * L_recon + lam_attr * L_attr + (1 / 13) * (
                    L_cluster_allmean + 0.1 * L_sep)

    return total_loss, L_pred, L_recon, L_attr, L_cluster_allmean, L_sep

def sep_loss(max_dist, selected_dists, indices):
        num_classes = len(indices)
        ranges = [list(range(1,num_classes))]
        for i in range(num_classes-2):
            ranges.append(list(range(0,num_classes-(num_classes-1-i)))+list(range(i+2,num_classes)))
        ranges.append(list(range(0,num_classes-1)))
        loss_temp = []
        for i in range(num_classes):
            loss_temp.append(torch.min(torch.maximum(torch.zeros_like(
                max_dist - selected_dists[indices[i],ranges[i]]),
                (max_dist - selected_dists[indices[i],ranges[i]])),
                dim=-1)[0])
        return torch.sum(torch.cat(loss_temp, dim=0))
