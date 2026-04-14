"""
Training functions

Author: Luisa Gallée, Github: `https://github.com/XRad-Ulm/HierViT`
"""

import torch
import wandb
import numpy as np
from sklearn.metrics import confusion_matrix
from loss import model_loss

def train_model(trainmodel, data_loader, args, epoch, optim, idx_with_attri):
    """
    Train model for one poch.
    :param trainmodel: model to be trained
    :param data_loader: data_loader used for training
    :param args: parser arguments
    :param epoch: current training epoch
    :param optim: training optimizer
    :param idx_with_attri: indices of samples used for attribute training
    :return: [trained model, mal_accuracy, attribute_accuracies]
    """
    trainmodel.train()
    torch.autograd.set_detect_anomaly(True)

    if args.dataset == "LIDC":
        num_attributes = 8
    elif args.dataset == "derm7pt":
        num_attributes = 7
    elif args.dataset == "FunnyNodules":
        num_attributes = 6

    correct_mal = 0
    correct_att = torch.zeros((num_attributes,))
    attrisamples = 0
    if args.dataset == "Chexbert":
        target_confusionmatrix = torch.zeros((2,2))
        attris_confusionmatrix = torch.zeros((num_attributes,2,2))

    for i, data in enumerate(data_loader):
        print(str(i)+" / "+str(len(data_loader)))
        if args.dataset == "LIDC":
            lam_attr = 1.0/8.0
            (x, y_mask, y_attributes, y_mal, sampleID, _) = data
            x, y_mask, y_attributes, y_mal = x.to("cuda", dtype=torch.float), y_mask.to("cuda", dtype=torch.float), \
                y_attributes.to("cuda", dtype=torch.float), y_mal.to("cuda", dtype=torch.float)
        elif args.dataset == "derm7pt":
            lam_attr = 1.0/7.0
            (x, y_mask, y_attributes, y_mal, sampleID) = data
            x, y_mask, y_attributes, y_mal = x.to("cuda", dtype=torch.float), y_mask.to("cuda", dtype=torch.float), \
                y_attributes.to("cuda", dtype=torch.int64), y_mal.to("cuda", dtype=torch.int64)
        elif args.dataset == "Chexbert":
            lam_attr = 1
            (x, y_mal, y_attributes, sampleID) = data
            y_mask = torch.tensor(0)
            x, y_mal, y_attributes, y_mask = x.to("cuda", dtype=torch.float), y_mal.to("cuda", dtype=torch.float), \
                y_attributes.to("cuda", dtype=torch.float), y_mask.to("cuda", dtype=torch.float)
        elif args.dataset == "FunnyNodules":
            lam_attr = 1.0/6.0
            (x, y_mask, y_attributes, y_mal, sampleID) = data
            x,y_mask, y_attributes, y_mal = x.to("cuda", dtype=torch.float), y_mask.to("cuda", dtype=torch.float),\
                y_attributes.to("cuda", dtype=torch.float), y_mal.to("cuda", dtype=torch.float)


        optim.zero_grad()
        if args.base_model in ["ResNetMT","DenseNetMT"]:
            x = x.repeat(1, 3, 1, 1)
            outputs = trainmodel(x)
            pred_mal = outputs[:, 0]
            pred_attr = outputs[:, 1:]
            loss, L_pred, L_recon, L_attr, L_cluster, L_sep = model_loss(y_mal, pred_mal, y_mask, y_mask,
                                                                         0,
                                                                         y_attributes,
                                                                         pred_attr,
                                                                         lam_attr=lam_attr, epoch=epoch,
                                                                         dists_to_protos=0,
                                                                         sample_id=sampleID,
                                                                         idx_with_attri=idx_with_attri,
                                                                         max_dist=0,
                                                                         args=args)
        elif args.base_model == "ViT":
            pred_outs, pred_recon = trainmodel(x)

            if args.dataset == "LIDC":
                pred_mal = pred_outs[:, 8:]
                pred_attr = pred_outs[:, :8]
            elif args.dataset == "derm7pt":
                pred_attr = []
                # 3,2,3,3,3,3,2
                pred_attr.append(pred_outs[:, 0:3])
                pred_attr.append(pred_outs[:, 3:5])
                pred_attr.append(pred_outs[:, 5:8])
                pred_attr.append(pred_outs[:, 8:11])
                pred_attr.append(pred_outs[:, 11:14])
                pred_attr.append(pred_outs[:, 14:17])
                pred_attr.append(pred_outs[:, 17:19])
                pred_mal = pred_outs[:, 19:]
            elif args.dataset == "FunnyNodules":
                pred_attr = pred_outs[:, :6]
                pred_mal = pred_outs[:, 6:]

            dists_to_protos = trainmodel.getDistance(x)

            loss, L_pred, L_recon, L_attr, L_cluster, L_sep = model_loss(y_mal, pred_mal, y_mask, pred_recon,
                                                                         args.lam_recon,
                                                                         y_attributes,
                                                                         pred_attr,
                                                                         lam_attr=lam_attr, epoch=epoch,
                                                                         dists_to_protos=dists_to_protos,
                                                                         sample_id=sampleID,
                                                                         idx_with_attri=idx_with_attri,
                                                                         max_dist=0,
                                                                         args=args)
        else:
            pred_mal, pred_attr, x_recon, dists_to_protos = trainmodel(x)
            if args.dataset == "derm7pt":
                pred_attr_new = []
                # 3,2,3,3,3,3,2
                pred_attr_new.append(pred_attr[:, 0:3])
                pred_attr_new.append(pred_attr[:, 3:5])
                pred_attr_new.append(pred_attr[:, 5:8])
                pred_attr_new.append(pred_attr[:, 8:11])
                pred_attr_new.append(pred_attr[:, 11:14])
                pred_attr_new.append(pred_attr[:, 14:17])
                pred_attr_new.append(pred_attr[:, 17:19])
                pred_attr = pred_attr_new

            loss, L_pred, L_recon, L_attr, L_cluster, L_sep = model_loss(y_mal, pred_mal, y_mask, x_recon,
                                                                         args.lam_recon,
                                                                         y_attributes,
                                                                         pred_attr,
                                                                         lam_attr=lam_attr, epoch=epoch,
                                                                         dists_to_protos=dists_to_protos,
                                                                         sample_id=sampleID,
                                                                         idx_with_attri=idx_with_attri,
                                                                         max_dist=trainmodel.out_dim_caps,
                                                                         args=args)

        wandb.log(
            {'train_loss': loss, 'train_loss_pred': L_pred, 'train_loss_recon': L_recon, 'train_loss_attr': L_attr,
             'epoch': epoch, 'train_loss_cluster': L_cluster, 'train_loos_sep': L_sep})

        loss.backward()
        optim.step()

        if args.dataset in ["LIDC", "derm7pt", "FunnyNodules"]:
            if len(pred_mal.shape) < 2:
                if not args.base_model=="ViT":
                    pred_mal = torch.unsqueeze(pred_mal, 0)

            if args.base_model=="ViT":
                if args.dataset == "derm7pt":
                    mal_confusion_matrix = confusion_matrix(y_mal.cpu().detach().numpy(),
                                                            torch.argmax(pred_mal,dim=-1).cpu().detach().numpy(),
                                                            labels=[0,1,2,3,4])

                elif args.dataset == "LIDC":
                    pred_mal *= 4
                    pred_mal = torch.round(pred_mal)
                    mal_confusion_matrix = confusion_matrix(np.argmax(y_mal.cpu().detach().numpy(), axis=1) + 1,
                                                            pred_mal.cpu().detach().numpy() + 1,
                                                            labels=[1, 2, 3, 4, 5])
                elif args.dataset == "FunnyNodules":
                    pred_mal *= 4
                    pred_mal = torch.round(pred_mal)
                    y_mal *= 4
                    mal_confusion_matrix = confusion_matrix(y_mal.cpu().detach().numpy() + 1,
                                                            torch.squeeze(pred_mal).cpu().detach().numpy() + 1,
                                                            labels=[1, 2, 3, 4, 5])
                from torchmetrics import Dice
                dice = Dice(average='micro').to("cuda")
                if args.dataset in ["LIDC", "FunnyNodules"]:
                    wandb.log({'train_dice_step': dice(pred_recon, y_mask.type(torch.int64))})
            elif args.dataset == "derm7pt":
                mal_confusion_matrix = confusion_matrix(y_mal.cpu().detach().numpy(),
                                                        torch.argmax(pred_mal, dim=-1).cpu().detach().numpy(),
                                                        labels=[0, 1, 2, 3, 4])
            elif args.dataset == "FunnyNodules":
                pred_mal *= 4
                pred_mal = torch.round(pred_mal)
                y_mal *= 4
                mal_confusion_matrix = confusion_matrix(y_mal.cpu().detach().numpy() + 1,
                                                        torch.squeeze(pred_mal).cpu().detach().numpy() + 1,
                                                        labels=[1, 2, 3, 4, 5])
            else:
                mal_confusion_matrix = confusion_matrix(np.argmax(y_mal.cpu().detach().numpy(), axis=1) + 1,
                                                        np.argmax(pred_mal.cpu().detach().numpy(), axis=1) + 1,
                                                        labels=[1, 2, 3, 4, 5])
            if args.dataset == "derm7pt":
                mal_correct_within_one = sum(np.diagonal(mal_confusion_matrix, offset=0))
            else:
                mal_correct_within_one = sum(np.diagonal(mal_confusion_matrix, offset=0)) + \
                                         sum(np.diagonal(mal_confusion_matrix, offset=1)) + \
                                         sum(np.diagonal(mal_confusion_matrix, offset=-1))
            correct_mal += mal_correct_within_one
            wandb.log({'train_acc_step': mal_correct_within_one})


            if args.dataset == "derm7pt":
                batchidx_with_attri = []
                for sai in range(len(sampleID)):
                    if sampleID[sai] > -1:
                        batchidx_with_attri.append(sai)
                attrisamples += len(batchidx_with_attri)
                for i in range(7):
                    correct_att[i] += torch.sum(y_attributes[batchidx_with_attri,i] == torch.argmax(pred_attr[i][batchidx_with_attri],dim=-1)).cpu().detach().numpy()

            elif args.dataset == "LIDC":
                y_attributes[:, [0, 3, 4, 5, 6, 7]] *= 4
                y_attributes[:, 1] *= 3
                y_attributes[:, 2] *= 5
                y_attributes += 1
                pred_attr[:, [0, 3, 4, 5, 6, 7]] *= 4
                pred_attr[:, 1] *= 3
                pred_attr[:, 2] *= 5
                pred_attr += 1
                for at in range(y_attributes.shape[1]):
                    a_labels = [1, 2, 3, 4, 5]
                    if num_attributes == 8:
                        if at == 1:
                            a_labels = [1, 2, 3, 4]
                        if at == 2:
                            a_labels = [1, 2, 3, 4, 5, 6]
                    attr_confusion_matrix = confusion_matrix(
                        np.rint(y_attributes[:, at].cpu().detach().numpy()),
                        np.rint(pred_attr[:, at].cpu().detach().numpy()),
                        labels=a_labels)
                    correct_att[at] += sum(np.diagonal(attr_confusion_matrix, offset=0)) + \
                                       sum(np.diagonal(attr_confusion_matrix, offset=1)) + \
                                       sum(np.diagonal(attr_confusion_matrix, offset=-1))
            elif args.dataset == "FunnyNodules":
                y_attributes[:, :-1] *= 4
                y_attributes[:, :-1] += 1
                pred_attr[:, :-1] *= 4
                pred_attr[:, :-1] += 1
                for at in range(y_attributes.shape[1]):
                    if at < 5:
                        a_labels = [1, 2, 3, 4, 5]
                        attr_confusion_matrix = confusion_matrix(
                            np.rint(y_attributes[:, at].cpu().detach().numpy()).astype(int),
                            np.rint(pred_attr[:, at].cpu().detach().numpy()).astype(int),
                            labels=a_labels)
                        correct_att[at] += sum(np.diagonal(attr_confusion_matrix, offset=0)) + \
                                           sum(np.diagonal(attr_confusion_matrix, offset=1)) + \
                                           sum(np.diagonal(attr_confusion_matrix, offset=-1))
                    else:
                        attr_confusion_matrix = confusion_matrix(
                            np.rint(y_attributes[:, at].cpu().detach().numpy()).astype(int),
                            np.rint(pred_attr[:, at].cpu().detach().numpy()).astype(int),
                            labels=[0,1])
                        correct_att[at] += sum(np.diagonal(attr_confusion_matrix, offset=0))


        elif args.dataset == "Chexbert":
            pred_confusion_matrix = confusion_matrix(
                y_mal.cpu().detach().numpy(),
                np.rint(pred_mal.cpu().detach().numpy()),
                labels=[0,1])
            target_confusionmatrix += pred_confusion_matrix
            mal_correct_within_one = sum(np.diagonal(pred_confusion_matrix, offset=0))
            correct_mal += mal_correct_within_one

            for at in range(y_attributes.shape[1]):
                attr_confusion_matrix = confusion_matrix(
                    np.rint(y_attributes[:, at].cpu().detach().numpy()),
                    np.rint(pred_attr[:, at].cpu().detach().numpy()),
                    labels=[0,1])
                attris_confusionmatrix[at] += attr_confusion_matrix
                correct_att[at] += sum(np.diagonal(attr_confusion_matrix, offset=0))


    if args.dataset == "LIDC":
        return trainmodel, correct_mal / len(data_loader.dataset), \
            [correct_att[0] / len(data_loader.dataset),
             correct_att[1] / len(data_loader.dataset),
             correct_att[2] / len(data_loader.dataset),
             correct_att[3] / len(data_loader.dataset),
             correct_att[4] / len(data_loader.dataset),
             correct_att[5] / len(data_loader.dataset),
             correct_att[6] / len(data_loader.dataset),
             correct_att[7] / len(data_loader.dataset)]
    elif args.dataset == "FunnyNodules":
        return trainmodel, correct_mal / len(data_loader.dataset), \
            [correct_att[0] / len(data_loader.dataset),
             correct_att[1] / len(data_loader.dataset),
             correct_att[2] / len(data_loader.dataset),
             correct_att[3] / len(data_loader.dataset),
             correct_att[4] / len(data_loader.dataset),
             correct_att[5] / len(data_loader.dataset)]
    elif args.dataset == "derm7pt":
        return trainmodel, correct_mal / len(data_loader.dataset), \
            [correct_att[0] / attrisamples,
             correct_att[1] / attrisamples,
             correct_att[2] / attrisamples,
             correct_att[3] / attrisamples,
             correct_att[4] / attrisamples,
             correct_att[5] / attrisamples,
             correct_att[6] / attrisamples]
    elif args.dataset == "Chexbert":
        target_accuracy = sum(np.diagonal(target_confusionmatrix, offset=0))/len(data_loader.dataset)
        target_precision = target_confusionmatrix[1,1]/(target_confusionmatrix[1,1]+target_confusionmatrix[1,0])
        target_recall = target_confusionmatrix[1,1]/(target_confusionmatrix[1,1]+target_confusionmatrix[0,1])
        target_f1 = 2*((target_precision*target_recall)/(target_precision+target_recall))
        return trainmodel, [target_accuracy,target_precision,target_recall,target_f1], \
            [correct_att[0] / len(data_loader.dataset),
             correct_att[1] / len(data_loader.dataset),
             correct_att[2] / len(data_loader.dataset),
             correct_att[3] / len(data_loader.dataset),
             correct_att[4] / len(data_loader.dataset),
             correct_att[5] / len(data_loader.dataset),
             correct_att[6] / len(data_loader.dataset),
             correct_att[7] / len(data_loader.dataset),
             correct_att[8] / len(data_loader.dataset),
             correct_att[9] / len(data_loader.dataset),
             correct_att[10] / len(data_loader.dataset),
             correct_att[11] / len(data_loader.dataset),
             correct_att[12] / len(data_loader.dataset)]
