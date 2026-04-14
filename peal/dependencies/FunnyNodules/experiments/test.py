"""
Testing functions

Author: Luisa Gallée, Github: `https://github.com/XRad-Ulm/HierViT`
"""

import os
import torch
import torchmetrics
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, balanced_accuracy_score
import seaborn as sns
from torchmetrics import Dice
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from loss import model_loss
from dataset_generator import create_attribute_dataloader

def test(testmodel, data_loader, args):
    """
    Test model
    :param testmodel: model to be tested
    :param data_loader: data_loader used for testing
    :param args: parser arguments
    :return: [tested model, mal_accuracy, attribute_accuracies]
    """
    testmodel.eval()
    test_loss_total = 0
    if args.dataset == "LIDC":
        num_attributes = 8
    elif args.dataset == "derm7pt":
        num_attributes = 7
    elif args.dataset == "FunnyNodules":
        num_attributes = 6
    correct_mal = 0
    correct_att = torch.zeros((num_attributes,))
    derm_mal_confusion_matrix = torch.zeros((5,5))
    derm_all_target_gt_labels = []
    derm_all_target_preds = []
    derm_attr_mal_confusion_matrixes = [torch.zeros((3,3)),torch.zeros((2,2)),torch.zeros((3,3)),torch.zeros((3,3)),torch.zeros((3,3)),torch.zeros((3,3)),torch.zeros((2,2))]
    derm_all_attr_gt_labels = [[],[],[],[],[],[],[]]
    derm_all_attr_preds = [[],[],[],[],[],[],[]]
    if args.dataset == "Chexbert":
        target_confusionmatrix = torch.zeros((2, 2))
        attris_confusionmatrix = torch.zeros((num_attributes, 2, 2))
    total_dice = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            print(str(i) + " / " + str(len(data_loader)))
            if args.dataset == "LIDC":
                (x, y_mask, y_attributes, y_mal, sampleID, _) = data
                x, y_mask, y_attributes, y_mal = x.to("cuda", dtype=torch.float), y_mask.to("cuda",
                                                                                            dtype=torch.float), \
                    y_attributes.to("cuda", dtype=torch.float), y_mal.to("cuda",
                                                                         dtype=torch.float)
            elif args.dataset == "derm7pt":
                (x, y_mask, y_attributes, y_mal, sampleID) = data
                x, y_mask, y_attributes, y_mal = x.to("cuda", dtype=torch.float), y_mask.to("cuda", dtype=torch.float), \
                    y_attributes.to("cuda", dtype=torch.int64), y_mal.to("cuda", dtype=torch.int64)
            elif args.dataset == "Chexbert":
                (x, y_mal, y_attributes, sampleID) = data
                y_mask = torch.tensor(0)
                x, y_mal, y_attributes, y_mask = x.to("cuda", dtype=torch.float), y_mal.to("cuda", dtype=torch.float), \
                    y_attributes.to("cuda", dtype=torch.float), y_mask.to("cuda", dtype=torch.float)
            elif args.dataset == "FunnyNodules":
                lam_attr = 1.0/6.0
                (x,y_mask, y_attributes, y_mal, sampleID) = data
                x,y_mask, y_attributes, y_mal = x.to("cuda", dtype=torch.float), y_mask.to("cuda", dtype=torch.float),\
                    y_attributes.to("cuda", dtype=torch.float), y_mal.to("cuda", dtype=torch.float)

                # print(y_attributes[0])
                # print(y_mal[0])
                # plt.imshow(torch.squeeze(x[0].cpu()).detach().numpy(), cmap="gray")
                # plt.show()
                # plt.imshow(torch.squeeze(y_mask[0].cpu()).detach().numpy(), cmap="gray")
                # plt.show()

            if args.base_model in ["ResNetMT","DenseNetMT"]:
                x = x.repeat(1, 3, 1, 1)
                outputs = testmodel(x)
                pred_mal = outputs[:, 0]
                pred_attr = outputs[:, 1:]
                loss, L_pred, L_recon, L_attr, L_cluster, L_sep = model_loss(y_mal, pred_mal, y_mask, y_mask,
                                                                             0,
                                                                             y_attributes,
                                                                             pred_attr,
                                                                             lam_attr=lam_attr, epoch=0,
                                                                             dists_to_protos=0,
                                                                             sample_id=sampleID,
                                                                             idx_with_attri=sampleID,
                                                                             max_dist=0,
                                                                             args=args)
            elif args.base_model == "ViT":
                pred_outs, pred_recon = testmodel(x)

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
                    pred_mal = pred_outs[:, 6:]
                    pred_attr = pred_outs[:, :6]
                dists_to_protos = testmodel.getDistance(x)

                loss, L_pred, L_recon, L_attr, L_cluster, L_sep = model_loss(y_mal, pred_mal, y_mask, pred_recon,
                                                                             args.lam_recon,
                                                                             y_attributes,
                                                                             pred_attr,
                                                                             lam_attr=1.0 / 8.0, epoch=0,
                                                                             dists_to_protos=dists_to_protos,
                                                                             sample_id=sampleID,
                                                                             idx_with_attri=sampleID,
                                                                             max_dist=16,
                                                                             args=args)
            else:
                pred_mal, pred_attr, pred_recon, dists_to_protos = testmodel(x)
                pred_mal = torch.squeeze(pred_mal)
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

                loss, L_pred, L_recon, L_attr, L_cluster, L_sep = model_loss(y_mal, pred_mal, y_mask, pred_recon,
                                                                             args.lam_recon, y_attributes,
                                                                             pred_attr,
                                                                             1.0 / 8.0, epoch=0,
                                                                             dists_to_protos=dists_to_protos,
                                                                             sample_id=sampleID,
                                                                             idx_with_attri=sampleID,
                                                                             max_dist=testmodel.out_dim_caps,
                                                                             args=args)
            test_loss_total += loss.item() * x.size(0)
            if args.dataset in ["LIDC", "derm7pt", "FunnyNodules"]:
                if len(pred_mal.shape) < 2:
                    if not args.base_model == "ViT":
                        pred_mal = torch.unsqueeze(pred_mal, 0)

                if args.base_model == "ViT":
                    if args.dataset == "derm7pt":
                        mal_confusion_matrix = confusion_matrix(y_mal.cpu().detach().numpy(),
                                                                torch.argmax(pred_mal,
                                                                             dim=-1).cpu().detach().numpy(),
                                                                labels=[0, 1, 2, 3, 4])
                    elif args.dataset == "LIDC":
                        pred_mal *= 4
                        pred_mal = torch.round(pred_mal)
                        mal_confusion_matrix = confusion_matrix(np.argmax(y_mal.cpu().detach().numpy(), axis=1) + 1,
                                                                pred_mal.cpu().detach().numpy() + 1,
                                                                labels=[1, 2, 3, 4, 5])
                    elif args.dataset == "FunnyNodules":
                        # print("in test mal")
                        # print(pred_mal)
                        # print(y_mal)
                        pred_mal *= 4
                        pred_mal = torch.round(pred_mal)
                        y_mal *= 4
                        mal_confusion_matrix = confusion_matrix(y_mal.cpu().detach().numpy() + 1,
                                                                torch.squeeze(pred_mal,-1).cpu().detach().numpy() + 1,
                                                                labels=[1, 2, 3, 4, 5])
                    if args.dataset in ["LIDC","FunnyNodules"]:
                        dice = Dice(average='micro').to("cuda")
                        total_dice += dice(pred_recon, y_mask.type(torch.int64))*pred_mal.shape[0]
                elif args.dataset == "derm7pt":
                    mal_confusion_matrix = confusion_matrix(y_mal.cpu().detach().numpy(),
                                                            torch.argmax(pred_mal, dim=-1).cpu().detach().numpy(),
                                                            labels=[0, 1, 2, 3, 4])
                elif args.dataset == "FunnyNodules":
                    pred_mal *= 4
                    pred_mal = torch.round(pred_mal)
                    y_mal *= 4
                    if len(pred_mal.shape)>1:
                        pred_mal = torch.squeeze(pred_mal,dim=0)
                    mal_confusion_matrix = confusion_matrix(y_mal.cpu().detach().numpy() + 1,
                                                            pred_mal.cpu().detach().numpy() + 1,
                                                            labels=[1, 2, 3, 4, 5])
                    if not args.base_model in ["ResNetMT","DenseNetMT"]:
                        dice = Dice(average='micro').to("cuda")
                        total_dice += dice(pred_recon, y_mask.type(torch.int64)) * y_mal.shape[0]
                    # print("in test")
                    # print(pred_mal.shape)
                    # print(dice(pred_recon, y_mask.type(torch.int64)))
                    # plt.imshow(torch.squeeze(pred_recon[0].cpu()).detach().numpy())
                    # plt.show()
                    # plt.imshow((torch.squeeze(y_mask[0].cpu())).detach().numpy())
                    # plt.show()
                else:
                    mal_confusion_matrix = confusion_matrix(np.argmax(y_mal.cpu().detach().numpy(), axis=1) + 1,
                                                            np.argmax(pred_mal.cpu().detach().numpy(), axis=1) + 1,
                                                            labels=[1, 2, 3, 4, 5])
                if args.dataset == "derm7pt":
                    mal_correct_within_one = sum(np.diagonal(mal_confusion_matrix, offset=0))
                    print("confusion matrix target")
                    print(mal_confusion_matrix)
                    if args.dataset == "derm7pt":
                        derm_mal_confusion_matrix += mal_confusion_matrix
                        derm_all_target_gt_labels.extend(list(y_mal.cpu().detach().numpy()))
                        derm_all_target_preds.extend(list(pred_mal.cpu().detach().numpy()))
                else:
                    mal_correct_within_one = sum(np.diagonal(mal_confusion_matrix, offset=0)) + \
                                             sum(np.diagonal(mal_confusion_matrix, offset=1)) + \
                                             sum(np.diagonal(mal_confusion_matrix, offset=-1))
                correct_mal += mal_correct_within_one


                if args.dataset == "derm7pt":
                    labels_all = [[0,1,2],[0,1],[0,1,2],[0,1,2],[0,1,2],[0,1,2],[0,1]]
                    for i in range(7):
                        correct_att[i] += torch.sum(
                            y_attributes[:, i] == torch.argmax(pred_attr[i], dim=-1)).cpu().detach().numpy()
                        derm_attr_mal_confusion_matrixes[i] += confusion_matrix(y_attributes[:, i].cpu().detach().numpy(),
                                                        torch.argmax(pred_attr[i], dim=-1).cpu().detach().numpy(),
                                                        labels=labels_all[i])
                        derm_all_attr_gt_labels[i].extend(list(y_attributes[:, i].cpu().detach().numpy()))
                        derm_all_attr_preds[i].extend(list(pred_attr[i].cpu().detach().numpy()))
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
                    # print("in test attr")
                    # print(y_attributes)
                    # print(pred_attr)
                    y_attributes[:,:-1] *= 4
                    y_attributes[:,:-1] += 1
                    pred_attr[:,:-1] *= 4
                    pred_attr[:,:-1] += 1
                    # print(y_attributes)
                    # print(pred_attr)
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
                                labels=[0, 1])
                            correct_att[at] += sum(np.diagonal(attr_confusion_matrix, offset=0))
            elif args.dataset == "Chexbert":
                pred_confusion_matrix = confusion_matrix(
                    y_mal.cpu().detach().numpy(),
                    np.rint(pred_mal.cpu().detach().numpy()),
                    labels=[0, 1])
                target_confusionmatrix += pred_confusion_matrix
                mal_correct_within_one = sum(np.diagonal(pred_confusion_matrix, offset=0))
                correct_mal += mal_correct_within_one

                for at in range(y_attributes.shape[1]):
                    attr_confusion_matrix = confusion_matrix(
                        np.rint(y_attributes[:, at].cpu().detach().numpy()),
                        np.rint(pred_attr[:, at].cpu().detach().numpy()),
                        labels=[0, 1])
                    attris_confusionmatrix[at] += attr_confusion_matrix
                    correct_att[at] += sum(np.diagonal(attr_confusion_matrix, offset=0))
    test_loss_total /= len(data_loader.dataset)
    if args.dataset in ["LIDC", "FunnyNodules"]:
        if not args.base_model in ["ResNetMT","DenseNetMT"]:
            print("Dice score mean: "+str(total_dice.item()/len(data_loader.dataset)))
    if args.dataset == "LIDC":
        return test_loss_total, correct_mal / len(data_loader.dataset), \
            [correct_att[0] / len(data_loader.dataset),
             correct_att[1] / len(data_loader.dataset),
             correct_att[2] / len(data_loader.dataset),
             correct_att[3] / len(data_loader.dataset),
             correct_att[4] / len(data_loader.dataset),
             correct_att[5] / len(data_loader.dataset),
             correct_att[6] / len(data_loader.dataset),
             correct_att[7] / len(data_loader.dataset)]
    elif args.dataset == "FunnyNodules":
        return test_loss_total, correct_mal / len(data_loader.dataset), \
            [correct_att[0] / len(data_loader.dataset),
             correct_att[1] / len(data_loader.dataset),
             correct_att[2] / len(data_loader.dataset),
             correct_att[3] / len(data_loader.dataset),
             correct_att[4] / len(data_loader.dataset),
             correct_att[5] / len(data_loader.dataset)]
    elif args.dataset == "derm7pt":
        print("bcc: "+str(balanced_accuracy_score(np.asarray(derm_all_target_gt_labels),np.argmax(np.asarray(derm_all_target_preds),axis=-1))))
        print(np.asarray(derm_all_target_gt_labels).shape)
        print(np.asarray(derm_all_target_preds).shape)
        print("mean roc auc: "+str(roc_auc_score(np.asarray(derm_all_target_gt_labels), np.asarray(derm_all_target_preds), multi_class='ovr')))
        classnames = ['nev','sk','misc','bcc','mel']
        class_colors = [(38/255,84/255,124/255),(125/255,154/255,170/255),(163/255,38/255,56/255),(86/255,170/255,28/255),(223/255,109/255,7/255)]
        font = {'family': 'Times new roman', 'size': 28}
        plt.figure(figsize=(8,8))
        ax_bottom = plt.subplot(1,1,1)
        for classidx in range(5):
            class_i = np.asarray([1 if y == classidx else 0 for y in np.asarray(derm_all_target_gt_labels)])
            prob_i = np.asarray(derm_all_target_preds)[:, classidx]
            print("roc_auc "+str(classnames[classidx])+": "+str(roc_auc_score(class_i, prob_i)))
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(class_i, prob_i)
            ax_bottom.plot(fpr, tpr, color=class_colors[classidx],lw=3)
        ax_bottom.set_ylabel("True Positive rate", fontdict=font)
        ax_bottom.set_xlabel("False Positive rate (100-Specificity)", fontdict=font)
        ax_bottom.legend(classnames, loc='lower right', prop=font)
        ax_bottom.plot([0, 1], [0, 1], color='black',ls='--')
        ax_bottom.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax_bottom.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax_bottom.set_xticklabels([0,0.2,0.4,0.6,0.8,1.0], fontdict=font)
        ax_bottom.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0], fontdict=font)
        plt.show()
        print("sensitivities "+str(classnames))
        sens_nev = (derm_mal_confusion_matrix[0,0]/torch.sum(derm_mal_confusion_matrix[0,:])).item()
        sens_sk = (derm_mal_confusion_matrix[1,1]/torch.sum(derm_mal_confusion_matrix[1,:])).item()
        sens_misc = (derm_mal_confusion_matrix[2,2]/torch.sum(derm_mal_confusion_matrix[2,:])).item()
        sens_bcc = (derm_mal_confusion_matrix[3,3]/torch.sum(derm_mal_confusion_matrix[3,:])).item()
        sens_mel = (derm_mal_confusion_matrix[4,4]/torch.sum(derm_mal_confusion_matrix[4,:])).item()
        print(str(sens_nev)+" "+str(sens_sk)+" "+str(sens_misc)+" "+str(sens_bcc)+" "+str(sens_mel))
        print("mean sensitivity: "+str(sum([sens_nev,sens_sk,sens_misc,sens_bcc,sens_mel])/5))
        print("specificity "+str(classnames))
        spec_nev = ((derm_mal_confusion_matrix[1,1]+derm_mal_confusion_matrix[2,2]+derm_mal_confusion_matrix[3,3]+derm_mal_confusion_matrix[4,4])/
              (derm_mal_confusion_matrix[1,1]+derm_mal_confusion_matrix[2,2]+derm_mal_confusion_matrix[3,3]+derm_mal_confusion_matrix[4,4]+torch.sum(derm_mal_confusion_matrix[[1,2,3,4],0]))).item()
        spec_sk = ((derm_mal_confusion_matrix[0,0] + derm_mal_confusion_matrix[2, 2] + derm_mal_confusion_matrix[3, 3] +
               derm_mal_confusion_matrix[4, 4]) /
              (derm_mal_confusion_matrix[0,0] + derm_mal_confusion_matrix[2, 2] + derm_mal_confusion_matrix[3, 3] +
               derm_mal_confusion_matrix[4, 4] + torch.sum(derm_mal_confusion_matrix[[0, 2, 3, 4], 1]))).item()
        spec_misc = ((derm_mal_confusion_matrix[0,0] + derm_mal_confusion_matrix[1,1] + derm_mal_confusion_matrix[3, 3] +
               derm_mal_confusion_matrix[4, 4]) /
              (derm_mal_confusion_matrix[0,0] + derm_mal_confusion_matrix[1,1] + derm_mal_confusion_matrix[3, 3] +
               derm_mal_confusion_matrix[4, 4] + torch.sum(derm_mal_confusion_matrix[[0, 1, 3, 4], 2]))).item()
        spec_bcc = ((derm_mal_confusion_matrix[0,0] + derm_mal_confusion_matrix[1,1] + derm_mal_confusion_matrix[2,2] +
               derm_mal_confusion_matrix[4, 4]) /
              (derm_mal_confusion_matrix[0,0] + derm_mal_confusion_matrix[1,1] + derm_mal_confusion_matrix[2,2] +
               derm_mal_confusion_matrix[4, 4] + torch.sum(derm_mal_confusion_matrix[[0,1,2, 4], 3]))).item()
        spec_mel = ((derm_mal_confusion_matrix[0,0] + derm_mal_confusion_matrix[1,1] + derm_mal_confusion_matrix[2,2] +
               derm_mal_confusion_matrix[3,3]) /
              (derm_mal_confusion_matrix[0,0] + derm_mal_confusion_matrix[1,1] + derm_mal_confusion_matrix[2,2] +
               derm_mal_confusion_matrix[3,3] + torch.sum(derm_mal_confusion_matrix[[0,1, 2, 3], 4]))).item()
        print(str(spec_nev) + " " + str(spec_sk) + " " + str(spec_misc) + " " + str(spec_bcc) + " " + str(spec_mel))
        print("mean specificity: " + str(sum([spec_nev, spec_sk, spec_misc, spec_bcc, spec_mel])/5))
        print("precision "+str(classnames))
        prec_nev = (derm_mal_confusion_matrix[0, 0] / torch.sum(derm_mal_confusion_matrix[:,0])).item()
        prec_sk = (derm_mal_confusion_matrix[1, 1] / torch.sum(derm_mal_confusion_matrix[:,1])).item()
        prec_misc = (derm_mal_confusion_matrix[2, 2] / torch.sum(derm_mal_confusion_matrix[:,2])).item()
        prec_bcc = (derm_mal_confusion_matrix[3, 3] / torch.sum(derm_mal_confusion_matrix[:,3])).item()
        prec_mel = (derm_mal_confusion_matrix[4, 4] / torch.sum(derm_mal_confusion_matrix[:,4])).item()
        print(str(prec_nev) + " " + str(prec_sk) + " " + str(prec_misc) + " " + str(prec_bcc) + " " + str(prec_mel))
        print("mean precision: " + str(sum([prec_nev, prec_sk, prec_misc, prec_bcc, prec_mel]) / 5))
        attr_names = ["pn","bmv","vs","pig","str","dag","rs"]
        attrs_classnum = [3,2,3,3,3,3,2]
        for attr_idx in range(len(derm_attr_mal_confusion_matrixes)):
            print("bcc: " + str(balanced_accuracy_score(np.asarray(derm_all_attr_gt_labels[attr_idx]),
                                                        np.argmax(np.asarray(derm_all_attr_preds[attr_idx]), axis=-1))))
            class_colors = [(38 / 255, 84 / 255, 124 / 255), (125 / 255, 154 / 255, 170 / 255),
                            (163 / 255, 38 / 255, 56 / 255)]
            font = {'family': 'Times new roman', 'size': 28}
            plt.figure(figsize=(8, 8))
            ax_bottom = plt.subplot(1, 1, 1)
            classnames = ["absent","regular","irregular"]
            if attr_idx == 0:
                classnames = ["absent", "typical", "atypical"]
            if attr_idx in [1,6]:
                classnames = ["absent", "present"]
            for classidx in range(attrs_classnum[attr_idx]):
                class_i = np.asarray([1 if y == classidx else 0 for y in np.asarray(derm_all_attr_gt_labels[attr_idx])])
                prob_i = np.asarray(derm_all_attr_preds[attr_idx])[:, classidx]
                print("roc_auc "+ str(attr_names[attr_idx]) + str(classidx) + ": " + str(roc_auc_score(class_i, prob_i)))
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(class_i, prob_i)
                ax_bottom.plot(fpr, tpr, color=class_colors[classidx], lw=3)
            ax_bottom.set_ylabel("True Positive rate", fontdict=font)
            ax_bottom.set_xlabel("False Positive rate (100-Specificity)", fontdict=font)
            ax_bottom.legend(classnames, loc='lower right', prop=font)
            ax_bottom.plot([0, 1], [0, 1], color='black', ls='--')
            ax_bottom.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax_bottom.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax_bottom.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontdict=font)
            ax_bottom.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontdict=font)
            plt.show()
            print("sensitivities " + str(attr_names[attr_idx]))
            sens_0 = (derm_attr_mal_confusion_matrixes[attr_idx][0, 0] / torch.sum(derm_attr_mal_confusion_matrixes[attr_idx][0, :])).item()
            sens_1 = (derm_attr_mal_confusion_matrixes[attr_idx][1, 1] / torch.sum(derm_attr_mal_confusion_matrixes[attr_idx][1, :])).item()
            allsens = [sens_0, sens_1]
            if derm_attr_mal_confusion_matrixes[attr_idx].shape[0] == 3:
                sens_2 = (derm_attr_mal_confusion_matrixes[attr_idx][2, 2] / torch.sum(derm_attr_mal_confusion_matrixes[attr_idx][2, :])).item()
                allsens.append(sens_2)
            print(allsens)
            print("mean sensitivity: " + str(sum(allsens) / len(allsens)))
            print("specificity " + str(attr_names[attr_idx]))
            if derm_attr_mal_confusion_matrixes[attr_idx].shape[0] == 2:
                spec_0 = ((derm_attr_mal_confusion_matrixes[attr_idx][1, 1]) /
                            (derm_attr_mal_confusion_matrixes[attr_idx][1, 1] + torch.sum(derm_attr_mal_confusion_matrixes[attr_idx][1, 0]))).item()
                spec_1 = ((derm_attr_mal_confusion_matrixes[attr_idx][0, 0]) /
                           (derm_attr_mal_confusion_matrixes[attr_idx][0, 0] + torch.sum(derm_attr_mal_confusion_matrixes[attr_idx][0, 1]))).item()
                print(str(spec_0) + " " + str(spec_1))
                print("mean specificity: " + str(sum([spec_0, spec_1]) / 2))
            elif derm_attr_mal_confusion_matrixes[attr_idx].shape[0] == 3:
                spec_0 = ((derm_attr_mal_confusion_matrixes[attr_idx][1, 1] + derm_attr_mal_confusion_matrixes[attr_idx][2, 2]) /
                          (derm_attr_mal_confusion_matrixes[attr_idx][1, 1] + derm_attr_mal_confusion_matrixes[attr_idx][2, 2] + torch.sum(
                                      derm_attr_mal_confusion_matrixes[attr_idx][[1, 2], 0]))).item()
                spec_1 = ((derm_attr_mal_confusion_matrixes[attr_idx][0, 0] + derm_attr_mal_confusion_matrixes[attr_idx][2, 2]) /
                          (derm_attr_mal_confusion_matrixes[attr_idx][0, 0] + derm_attr_mal_confusion_matrixes[attr_idx][2, 2] + torch.sum(
                                      derm_attr_mal_confusion_matrixes[attr_idx][[0, 2], 1]))).item()
                spec_2 = ((derm_attr_mal_confusion_matrixes[attr_idx][0, 0] + derm_attr_mal_confusion_matrixes[attr_idx][1, 1]) /
                          (derm_attr_mal_confusion_matrixes[attr_idx][0, 0] + derm_attr_mal_confusion_matrixes[attr_idx][1, 1] + torch.sum(
                                         derm_attr_mal_confusion_matrixes[attr_idx][[0, 1], 2]))).item()
                print(str(spec_0) + " " + str(spec_1) + " " + str(spec_2))
                print("mean specificity: " + str(sum([spec_0, spec_1, spec_2]) / 3))
            print("precision " + str(attr_names[attr_idx]))
            prec_0 = (derm_attr_mal_confusion_matrixes[attr_idx][0, 0] / torch.sum(derm_attr_mal_confusion_matrixes[attr_idx][:, 0])).item()
            prec_1 = (derm_attr_mal_confusion_matrixes[attr_idx][1, 1] / torch.sum(derm_attr_mal_confusion_matrixes[attr_idx][:, 1])).item()
            allprec = [prec_0, prec_1]
            if derm_attr_mal_confusion_matrixes[attr_idx].shape[0] == 3:
                prec_2 = (derm_attr_mal_confusion_matrixes[attr_idx][2, 2] / torch.sum(derm_attr_mal_confusion_matrixes[attr_idx][:, 2])).item()
                allprec.append(prec_2)
            print(allprec)
            print("mean precision: " + str(sum(allprec) / len(allprec)))

        return test_loss_total, correct_mal / len(data_loader.dataset), \
            [correct_att[0] / len(data_loader.dataset),
             correct_att[1] / len(data_loader.dataset),
             correct_att[2] / len(data_loader.dataset),
             correct_att[3] / len(data_loader.dataset),
             correct_att[4] / len(data_loader.dataset),
             correct_att[5] / len(data_loader.dataset),
             correct_att[6] / len(data_loader.dataset)]
    elif args.dataset == "Chexbert":
        target_accuracy = sum(np.diagonal(target_confusionmatrix, offset=0)) / len(data_loader.dataset)
        target_precision = target_confusionmatrix[1, 1] / (target_confusionmatrix[1, 1] + target_confusionmatrix[1, 0])
        target_recall = target_confusionmatrix[1, 1] / (target_confusionmatrix[1, 1] + target_confusionmatrix[0, 1])
        target_f1 = 2 * ((target_precision * target_recall) / (target_precision + target_recall))
        return test_loss_total, [target_accuracy, target_precision, target_recall, target_f1], \
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

def test_show_inference(testmodel, data_loader, epoch,
                        prototypefoldername, args):
    """
    Show results of inference
    :param testmodel: model to be tested
    :param test_loader: data loader used for testing
    :param epoch: number of epoch the model is being tested
    :param prototypefoldername: folder direction of prototypes
    """
    with (((torch.no_grad()))):
        for i, data in enumerate(data_loader):
            if args.dataset == "LIDC":
                (x, y_mask, y_attributes, y_mal, _, _) = data
                x, y_mask, y_attributes, y_mal = x.to("cuda", dtype=torch.float), y_mask.to("cuda",
                                                                                            dtype=torch.float), \
                    y_attributes.to("cuda", dtype=torch.float), y_mal.to("cuda",
                                                                         dtype=torch.float)
            elif args.dataset == "derm7pt":
                (x, y_mask, y_attributes, y_mal, sampleID) = data
                x, y_mask, y_attributes, y_mal = x.to("cuda", dtype=torch.float), y_mask.to("cuda", dtype=torch.float), \
                    y_attributes.to("cuda", dtype=torch.int64), y_mal.to("cuda", dtype=torch.int64)
            elif args.dataset == "FunnyNodules":
                (x, y_mask, y_attributes, y_mal, sampleID) = data
                x,y_mask, y_attributes, y_mal = x.to("cuda", dtype=torch.float), y_mask.to("cuda", dtype=torch.float),\
                    y_attributes.to("cuda", dtype=torch.float), y_mal.to("cuda", dtype=torch.float)
            else:
                return
            if args.base_model == "ViT":
                dists_to_protos = testmodel.getDistance(x)
                pred_outs, pred_mask, list_vit_attentions = testmodel.forward_plot(x)
                if args.dataset == "LIDC":
                    pred_mal = pred_outs[:, 8:]
                elif args.dataset == "derm7pt":
                    pred_mal = pred_outs[:, 19:]
                elif args.dataset == "FunnyNodules":
                    pred_mal = pred_outs[:, 6:]
            else:
                return

            for sai in range(pred_mal.shape[0]):
                # select sample that is correctly/wrongly predicted, has a specific malignancy class, ...
                condition = False
                if args.dataset == "LIDC":
                    thissamplepred = torch.round(pred_mal[sai]*4)
                    if abs(thissamplepred - torch.argmax(y_mal[sai],dim=-1).item()) < 2:
                        condition = True
                elif args.dataset == "derm7pt":
                    thissamplepred = torch.argmax(pred_mal[sai],dim=-1)
                    if thissamplepred != y_mal[sai]:
                        condition = True
                elif args.dataset == "FunnyNodules":
                    thissamplepred = torch.round(pred_mal[sai]*4)
                    thissamplegt = torch.round(y_mal[sai]*4)
                    if thissamplepred == thissamplegt:
                        condition = True
                if condition:
                    _, min_proto_idx0 = torch.min(torch.flatten(dists_to_protos[0], start_dim=1), dim=-1)
                    min_proto_idx0 = [unravel_index(i, dists_to_protos[0][sai].shape) for i in min_proto_idx0]
                    _, min_proto_idx1 = torch.min(torch.flatten(dists_to_protos[1], start_dim=1), dim=-1)
                    min_proto_idx1 = [unravel_index(i, dists_to_protos[1][sai].shape) for i in min_proto_idx1]
                    _, min_proto_idx2 = torch.min(torch.flatten(dists_to_protos[2], start_dim=1), dim=-1)
                    min_proto_idx2 = [unravel_index(i, dists_to_protos[2][sai].shape) for i in min_proto_idx2]
                    _, min_proto_idx3 = torch.min(torch.flatten(dists_to_protos[3], start_dim=1), dim=-1)
                    min_proto_idx3 = [unravel_index(i, dists_to_protos[3][sai].shape) for i in min_proto_idx3]
                    _, min_proto_idx4 = torch.min(torch.flatten(dists_to_protos[4], start_dim=1), dim=-1)
                    min_proto_idx4 = [unravel_index(i, dists_to_protos[4][sai].shape) for i in min_proto_idx4]
                    _, min_proto_idx5 = torch.min(torch.flatten(dists_to_protos[5], start_dim=1), dim=-1)
                    min_proto_idx5 = [unravel_index(i, dists_to_protos[5][sai].shape) for i in min_proto_idx5]
                    if args.dataset in ["LIDC","derm7pt"]:
                        _, min_proto_idx6 = torch.min(torch.flatten(dists_to_protos[6], start_dim=1), dim=-1)
                        min_proto_idx6 = [unravel_index(i, dists_to_protos[6][sai].shape) for i in min_proto_idx6]
                    if args.dataset == "LIDC":
                        _, min_proto_idx7 = torch.min(torch.flatten(dists_to_protos[7], start_dim=1), dim=-1)
                        min_proto_idx7 = [unravel_index(i, dists_to_protos[7][sai].shape) for i in min_proto_idx7]
                        min_proto_idx_allcaps = [min_proto_idx0, min_proto_idx1, min_proto_idx2, min_proto_idx3,
                                                 min_proto_idx4, min_proto_idx5, min_proto_idx6, min_proto_idx7]
                    elif args.dataset == "derm7pt":
                        min_proto_idx_allcaps = [min_proto_idx0, min_proto_idx1, min_proto_idx2, min_proto_idx3,
                                                 min_proto_idx4, min_proto_idx5, min_proto_idx6]
                    elif args.dataset == "FunnyNodules":
                        min_proto_idx_allcaps = [min_proto_idx0, min_proto_idx1, min_proto_idx2, min_proto_idx3,
                                                 min_proto_idx4, min_proto_idx5]

                    folder_dir = "./prototypes/" + str(prototypefoldername) + "/" + str(epoch)
                    if args.dataset == "LIDC":
                        fig, axes = plt.subplots(nrows=3, ncols=8)
                    elif args.dataset == "derm7pt":
                        fig, axes = plt.subplots(nrows=3, ncols=7)
                    elif args.dataset == "FunnyNodules":
                        fig, axes = plt.subplots(nrows=3, ncols=6)
                    fig.set_size_inches(24,12)
                    if args.dataset in ["LIDC","FunnyNodules"]:
                        axes.flat[0].imshow(torch.squeeze(x[sai]).cpu(), cmap='gray')
                    elif args.dataset == "derm7pt":
                        axes.flat[0].imshow(torch.squeeze(x[sai].permute(1,2,0)).cpu())
                    axes.flat[0].set_title("input")

                    thistaratt = (list_vit_attentions[sai][-1]-np.min(list_vit_attentions[sai][-1]))/(np.max(list_vit_attentions[sai][-1])-np.min(list_vit_attentions[sai][-1]))
                    tar_sum = axes.flat[1].imshow(thistaratt, cmap="YlGnBu")
                    fig.colorbar(tar_sum, ax=axes.flat[1])
                    axes.flat[1].set_title("vit_att_tar")

                    if args.dataset in ["LIDC","FunnyNodules"]:
                        axes.flat[3].imshow(torch.squeeze(y_mask[sai]).cpu(), cmap='gray')
                        axes.flat[3].set_title("mask")
                        axes.flat[4].imshow(torch.squeeze(pred_mask[sai]).cpu(), cmap='gray')
                        axes.flat[4].set_title("mask pred")
                    if args.dataset == "LIDC":
                        numattris = 8
                        counter = 8
                        cpsnames = ["sub", "is", "cal", "sph", "mar", "lob", "spic", "tex"]
                    elif args.dataset == "derm7pt":
                        numattris = 7
                        counter = 7
                        cpsnames = ["pn","bmv","vs","pig","str","dag","rs"]
                    elif args.dataset == "FunnyNodules":
                        numattris = 6
                        counter = 6
                        cpsnames = ["roundness","spiculation","edge_sharpness","size","intensity","internal_structure"]
                    x_ex = torch.zeros((x.shape[0], len(min_proto_idx_allcaps), testmodel.protodigis0.shape[-2],
                                        testmodel.protodigis0.shape[-1]))

                    for capsule_idx in range(len(min_proto_idx_allcaps)):
                        x_ex[sai, capsule_idx, :] = testmodel.protodigis_list[capsule_idx][
                            min_proto_idx_allcaps[capsule_idx][sai][0], min_proto_idx_allcaps[capsule_idx][sai][1]]

                        thisattriatt = (list_vit_attentions[sai][capsule_idx+2]-np.min(list_vit_attentions[sai][capsule_idx+2]))/(np.max(list_vit_attentions[sai][capsule_idx+2])-np.min(list_vit_attentions[sai][capsule_idx+2]))
                        tar_sum = axes.flat[counter+numattris].imshow(thisattriatt, cmap="YlGnBu")
                        fig.colorbar(tar_sum, ax=axes.flat[counter+numattris])
                        axes.flat[counter+numattris].set_title("vit_att_"+str(capsule_idx))

                        for image_name in os.listdir(folder_dir):
                            if image_name.startswith(
                                    "cpslnr" + str(capsule_idx) + "_protonr" + str(
                                        min_proto_idx_allcaps[capsule_idx][sai][0]) + "-" + str(
                                        min_proto_idx_allcaps[capsule_idx][sai][1]) + "_"):
                                proto_attrsc_str = image_name.split("[")[1].split("]")[0].split(", ")
                                protoimage = np.load(
                                    folder_dir + "/" + image_name)
                                axes.flat[counter].imshow(protoimage, cmap='gray')
                                axes.flat[counter].set_title(cpsnames[capsule_idx] + " gt: " + str(
                                        round(y_attributes[sai][capsule_idx].item(), 2)) + ",proto: " + str(
                                        round(float(proto_attrsc_str[capsule_idx]), 2)))
                                counter += 1

                    pred_p = testmodel.forwardprotodigis(x_ex=x_ex.to("cuda", dtype=torch.float))
                    if args.dataset == "LIDC":
                        plt.suptitle("ytrue:"+str(torch.argmax(y_mal[sai], dim=-1).item())+"ypred:"+str(thissamplepred.item())+" yproto:"+str(torch.round(pred_p[sai]*4).item()))
                    elif args.dataset == "derm7pt":
                        plt.suptitle("ytrue:"+str(y_mal[sai].item())+"ypred:"+str(thissamplepred.item())+" yproto:"+str(torch.argmax(pred_p[sai], dim=-1).item()))
                    elif args.dataset == "FunnyNodules":
                        plt.suptitle("ytrue:"+str(thissamplegt.item()+1)+"ypred:"+str(thissamplepred.item()+1)+str(torch.round(pred_p[sai]*4).item()+1))
                    if args.dataset == "LIDC":
                        thissamplepred = torch.round(pred_p[sai] * 4)
                        if abs(thissamplepred - torch.argmax(y_mal[sai],dim=-1).item()) < 2 :
                            plt.close(fig)
                            continue
                    for i in range(3*numattris):
                        axes.flat[i].axis('off')
                    plt.show()

def test_indepth_attripredCorr(testmodel, train_loader, test_loader, epoch,
                               prototypefoldername, args):
    """
    Test model regarding correlation of correctness of attribute and malignancy testing
    :param testmodel: model to be tested
    :param train_loader: data loader used for training
    :param test_loader: data loader used for testing
    :param epoch: number of epoch the model is being tested
    :param prototypefoldername: folder direction of prototypes
    """
    testmodel.eval()
    predp_attrip_correct_matrix = np.zeros(((8, 2, 2)))
    #                   predp
    #                   correct  false
    # attrip   correct
    #          false
    num_attributes = next(iter(test_loader))[2].shape[1]
    attricorr_train = torch.zeros((len(train_loader.dataset), num_attributes))
    attricorr_test = torch.zeros((len(test_loader.dataset), num_attributes))
    predcorr_train = torch.zeros((len(train_loader.dataset)))
    predcorr_test = torch.zeros((len(test_loader.dataset)))
    batch_num = -1
    batch_size = next(iter(test_loader))[0].shape[0]
    with torch.no_grad():
        for x, y_mask, y_attributes, y_mal, _ in test_loader:
            batch_num += 1
            x, y_mask, y_attributes, y_mal = x.to("cuda", dtype=torch.float), y_mask.to("cuda", dtype=torch.float), \
                y_attributes.to("cuda", dtype=torch.float), y_mal.to("cuda",
                                                                     dtype=torch.float)

            _, _, pred_mask, dists_to_protos = testmodel(x)

            _, min_proto_idx0 = torch.min(torch.flatten(dists_to_protos[0], start_dim=1), dim=-1)
            print(min_proto_idx0.shape)
            min_proto_idx0 = [unravel_index(i, dists_to_protos[0][0].shape) for i in min_proto_idx0]
            print(len(min_proto_idx0))
            print(len(min_proto_idx0[0]))
            _, min_proto_idx1 = torch.min(torch.flatten(dists_to_protos[1], start_dim=1), dim=-1)
            min_proto_idx1 = [unravel_index(i, dists_to_protos[1][0].shape) for i in min_proto_idx1]
            _, min_proto_idx2 = torch.min(torch.flatten(dists_to_protos[2], start_dim=1), dim=-1)
            min_proto_idx2 = [unravel_index(i, dists_to_protos[2][0].shape) for i in min_proto_idx2]
            _, min_proto_idx3 = torch.min(torch.flatten(dists_to_protos[3], start_dim=1), dim=-1)
            min_proto_idx3 = [unravel_index(i, dists_to_protos[3][0].shape) for i in min_proto_idx3]
            _, min_proto_idx4 = torch.min(torch.flatten(dists_to_protos[4], start_dim=1), dim=-1)
            min_proto_idx4 = [unravel_index(i, dists_to_protos[4][0].shape) for i in min_proto_idx4]
            _, min_proto_idx5 = torch.min(torch.flatten(dists_to_protos[5], start_dim=1), dim=-1)
            min_proto_idx5 = [unravel_index(i, dists_to_protos[5][0].shape) for i in min_proto_idx5]
            _, min_proto_idx6 = torch.min(torch.flatten(dists_to_protos[6], start_dim=1), dim=-1)
            min_proto_idx6 = [unravel_index(i, dists_to_protos[6][0].shape) for i in min_proto_idx6]
            _, min_proto_idx7 = torch.min(torch.flatten(dists_to_protos[7], start_dim=1), dim=-1)
            min_proto_idx7 = [unravel_index(i, dists_to_protos[7][0].shape) for i in min_proto_idx7]

            min_proto_idx_allcaps = [min_proto_idx0, min_proto_idx1, min_proto_idx2, min_proto_idx3,
                                     min_proto_idx4, min_proto_idx5, min_proto_idx6, min_proto_idx7]

            x_ex = torch.zeros((x.shape[0], len(min_proto_idx_allcaps), testmodel.protodigis0.shape[-1]))

            attrLabelsSamples_protos = torch.zeros_like(y_attributes)
            folder_dir = "./prototypes/" + str(prototypefoldername) + "/" + str(epoch)

            for sai in range(x.shape[0]):
                for capsule_idx in range(len(min_proto_idx_allcaps)):

                    x_ex[sai, capsule_idx, :] = testmodel.protodigis_list[capsule_idx][
                        min_proto_idx_allcaps[capsule_idx][sai][0], min_proto_idx_allcaps[capsule_idx][sai][1]]
                    for image_name in os.listdir(folder_dir):
                        if image_name.startswith(
                                "cpslnr" + str(capsule_idx) + "_protonr" + str(
                                    min_proto_idx_allcaps[capsule_idx][sai][0]) + "-" + str(
                                    min_proto_idx_allcaps[capsule_idx][sai][1]) + "_"):
                            proto_attrsc_str = image_name.split("[")[1].split("]")[0].split(", ")
                            proto_attrsc = torch.tensor(np.array(proto_attrsc_str).astype(np.float64))
                            attrLabelsSamples_protos[sai, capsule_idx] = proto_attrsc[capsule_idx]

            pred_p, _, _ = testmodel.forwardCapsule(x_ex=x_ex.to("cuda", dtype=torch.float))
            pred_p_correct_0 = np.argmax(y_mal.cpu().detach().numpy(), axis=1) == np.argmax(
                pred_p.cpu().detach().numpy(), axis=1)
            pred_p_correct_p1 = np.argmax(y_mal.cpu().detach().numpy(), axis=1) + 1 == np.argmax(
                pred_p.cpu().detach().numpy(), axis=1)
            pred_p_correct_m1 = np.argmax(y_mal.cpu().detach().numpy(), axis=1) - 1 == np.argmax(
                pred_p.cpu().detach().numpy(), axis=1)
            pred_p_correct_list = pred_p_correct_0 + pred_p_correct_p1 + pred_p_correct_m1
            predcorr_test[
            (batch_num * batch_size):(batch_num * batch_size + len(pred_p_correct_list))] = torch.from_numpy(
                pred_p_correct_list)

            y_attributes[:, [0, 3, 4, 5, 6, 7]] *= 4
            y_attributes[:, 1] *= 3
            y_attributes[:, 2] *= 5
            y_attributes += 1
            attrLabelsSamples_protos[:, [0, 3, 4, 5, 6, 7]] *= 4
            attrLabelsSamples_protos[:, 1] *= 3
            attrLabelsSamples_protos[:, 2] *= 5
            attrLabelsSamples_protos += 1
            all_attri_correct = []
            for at in range(y_attributes.shape[1]):
                attri_p_correct_0 = np.rint(y_attributes[:, at].cpu().detach().numpy()) == np.rint(
                    attrLabelsSamples_protos[:, at].cpu().detach().numpy())
                attri_p_correct_p1 = np.rint(y_attributes[:, at].cpu().detach().numpy()) + 1 == np.rint(
                    attrLabelsSamples_protos[:, at].cpu().detach().numpy())
                attri_p_correct_m1 = np.rint(y_attributes[:, at].cpu().detach().numpy()) - 1 == np.rint(
                    attrLabelsSamples_protos[:, at].cpu().detach().numpy())
                attri_p_correct_list = attri_p_correct_0 + attri_p_correct_p1 + attri_p_correct_m1
                all_attri_correct.append(attri_p_correct_list)
                attricorr_test[(batch_num * batch_size):(batch_num * batch_size + len(attri_p_correct_list)),
                at] = torch.from_numpy(attri_p_correct_list)

            for sai in range(x.shape[0]):
                for attri in range(len(all_attri_correct)):
                    if (pred_p_correct_list[sai] == True) and (all_attri_correct[attri][sai] == True):
                        predp_attrip_correct_matrix[attri, 0, 0] += 1
                    if (pred_p_correct_list[sai] == False) and (all_attri_correct[attri][sai] == True):
                        predp_attrip_correct_matrix[attri, 0, 1] += 1
                    if (pred_p_correct_list[sai] == True) and (all_attri_correct[attri][sai] == False):
                        predp_attrip_correct_matrix[attri, 1, 0] += 1
                    if (pred_p_correct_list[sai] == False) and (all_attri_correct[attri][sai] == False):
                        predp_attrip_correct_matrix[attri, 1, 1] += 1
    print("False attribute -> false mal prediction corrleation")
    cpsnames = ["sub", "is", "cal", "sph", "mar", "lob", "spic", "tex"]
    print("\t\t\t\t\t\t\t\tmal prediction")
    print("\t\t\t\t\t\t\t\tcorrect\tfalse")
    print("attri prediction\tcorrect")
    print("\t\t\t\t\tfalse")
    for attri in range(predp_attrip_correct_matrix.shape[0]):
        print(cpsnames[attri] + ":" + "\t\t\t\t\t\t\t" + str(predp_attrip_correct_matrix[attri][0, 0]) + "\t" + str(
            predp_attrip_correct_matrix[attri][0, 1]))
        print("\t\t\t\t\t\t\t\t" + str(predp_attrip_correct_matrix[attri][1, 0]) + "\t" + str(
            predp_attrip_correct_matrix[attri][1, 1]))
        print("FF/False attribute ratio: " + str(
            predp_attrip_correct_matrix[attri][1, 1] / np.sum(predp_attrip_correct_matrix[attri], axis=1)[1]))

    # train
    batch_num = -1
    batch_size = next(iter(train_loader))[0].shape[0]
    with torch.no_grad():
        for x, y_mask, y_attributes, y_mal, _ in train_loader:
            print(batch_num)
            batch_num += 1
            x, y_mask, y_attributes, y_mal = x.to("cuda", dtype=torch.float), y_mask.to("cuda", dtype=torch.float), \
                y_attributes.to("cuda", dtype=torch.float), y_mal.to("cuda",
                                                                     dtype=torch.float)

            _, _, pred_mask, dists_to_protos = testmodel(x)

            _, min_proto_idx0 = torch.min(torch.flatten(dists_to_protos[0], start_dim=1), dim=-1)
            min_proto_idx0 = [unravel_index(i, dists_to_protos[0][0].shape) for i in min_proto_idx0]
            _, min_proto_idx1 = torch.min(torch.flatten(dists_to_protos[1], start_dim=1), dim=-1)
            min_proto_idx1 = [unravel_index(i, dists_to_protos[1][0].shape) for i in min_proto_idx1]
            _, min_proto_idx2 = torch.min(torch.flatten(dists_to_protos[2], start_dim=1), dim=-1)
            min_proto_idx2 = [unravel_index(i, dists_to_protos[2][0].shape) for i in min_proto_idx2]
            _, min_proto_idx3 = torch.min(torch.flatten(dists_to_protos[3], start_dim=1), dim=-1)
            min_proto_idx3 = [unravel_index(i, dists_to_protos[3][0].shape) for i in min_proto_idx3]
            _, min_proto_idx4 = torch.min(torch.flatten(dists_to_protos[4], start_dim=1), dim=-1)
            min_proto_idx4 = [unravel_index(i, dists_to_protos[4][0].shape) for i in min_proto_idx4]
            _, min_proto_idx5 = torch.min(torch.flatten(dists_to_protos[5], start_dim=1), dim=-1)
            min_proto_idx5 = [unravel_index(i, dists_to_protos[5][0].shape) for i in min_proto_idx5]
            _, min_proto_idx6 = torch.min(torch.flatten(dists_to_protos[6], start_dim=1), dim=-1)
            min_proto_idx6 = [unravel_index(i, dists_to_protos[6][0].shape) for i in min_proto_idx6]
            _, min_proto_idx7 = torch.min(torch.flatten(dists_to_protos[7], start_dim=1), dim=-1)
            min_proto_idx7 = [unravel_index(i, dists_to_protos[7][0].shape) for i in min_proto_idx7]
            min_proto_idx_allcaps = [min_proto_idx0, min_proto_idx1, min_proto_idx2, min_proto_idx3,
                                     min_proto_idx4, min_proto_idx5, min_proto_idx6, min_proto_idx7]

            x_ex = torch.zeros((x.shape[0], len(min_proto_idx_allcaps), testmodel.protodigis0.shape[-1]))

            attrLabelsSamples_protos = torch.zeros_like(y_attributes)
            folder_dir = "./prototypes/" + str(prototypefoldername) + "/" + str(epoch)
            for sai in range(x.shape[0]):
                for capsule_idx in range(len(min_proto_idx_allcaps)):

                    x_ex[sai, capsule_idx, :] = testmodel.protodigis_list[capsule_idx][
                        min_proto_idx_allcaps[capsule_idx][sai][0], min_proto_idx_allcaps[capsule_idx][sai][1]]
                    for image_name in os.listdir(folder_dir):
                        if image_name.startswith(
                                "cpslnr" + str(capsule_idx) + "_protonr" + str(
                                    min_proto_idx_allcaps[capsule_idx][sai][0]) + "-" + str(
                                    min_proto_idx_allcaps[capsule_idx][sai][1]) + "_"):
                            proto_attrsc_str = image_name.split("[")[1].split("]")[0].split(", ")
                            proto_attrsc = torch.tensor(np.array(proto_attrsc_str).astype(np.float64))
                            attrLabelsSamples_protos[sai, capsule_idx] = proto_attrsc[capsule_idx]

            pred_p, _, _ = testmodel.forwardCapsule(x_ex=x_ex.to("cuda", dtype=torch.float))
            pred_p_correct_0 = np.argmax(y_mal.cpu().detach().numpy(), axis=1) == np.argmax(
                pred_p.cpu().detach().numpy(), axis=1)
            pred_p_correct_p1 = np.argmax(y_mal.cpu().detach().numpy(), axis=1) + 1 == np.argmax(
                pred_p.cpu().detach().numpy(), axis=1)
            pred_p_correct_m1 = np.argmax(y_mal.cpu().detach().numpy(), axis=1) - 1 == np.argmax(
                pred_p.cpu().detach().numpy(), axis=1)
            pred_p_correct_list = pred_p_correct_0 + pred_p_correct_p1 + pred_p_correct_m1
            predcorr_train[
            (batch_num * batch_size):(batch_num * batch_size + len(pred_p_correct_list))] = torch.from_numpy(
                pred_p_correct_list)

            y_attributes[:, [0, 3, 4, 5, 6, 7]] *= 4
            y_attributes[:, 1] *= 3
            y_attributes[:, 2] *= 5
            y_attributes += 1
            attrLabelsSamples_protos[:, [0, 3, 4, 5, 6, 7]] *= 4
            attrLabelsSamples_protos[:, 1] *= 3
            attrLabelsSamples_protos[:, 2] *= 5
            attrLabelsSamples_protos += 1
            all_attri_correct = []
            for at in range(y_attributes.shape[1]):
                attri_p_correct_0 = np.rint(y_attributes[:, at].cpu().detach().numpy()) == np.rint(
                    attrLabelsSamples_protos[:, at].cpu().detach().numpy())
                attri_p_correct_p1 = np.rint(y_attributes[:, at].cpu().detach().numpy()) + 1 == np.rint(
                    attrLabelsSamples_protos[:, at].cpu().detach().numpy())
                attri_p_correct_m1 = np.rint(y_attributes[:, at].cpu().detach().numpy()) - 1 == np.rint(
                    attrLabelsSamples_protos[:, at].cpu().detach().numpy())
                attri_p_correct_list = attri_p_correct_0 + attri_p_correct_p1 + attri_p_correct_m1
                all_attri_correct.append(attri_p_correct_list)
                attricorr_train[(batch_num * batch_size):(batch_num * batch_size + len(attri_p_correct_list)),
                at] = torch.from_numpy(attri_p_correct_list)
    clflr = LogisticRegression(random_state=0).fit(attricorr_train, predcorr_train)
    print("LogReg train" + str(clflr.score(attricorr_train, predcorr_train)))
    print("LogReg test" + str(clflr.score(attricorr_test, predcorr_test)))
    clfrf = RandomForestClassifier(max_depth=2, random_state=0).fit(attricorr_train, predcorr_train)
    print("RandFor train" + str(clfrf.score(attricorr_train, predcorr_train)))
    print("RandFor test" + str(clfrf.score(attricorr_test, predcorr_test)))
    clfdt = DecisionTreeClassifier(random_state=0).fit(attricorr_train, predcorr_train)
    print("DecTree train" + str(clfdt.score(attricorr_train, predcorr_train)))
    print("DecTree test" + str(clfdt.score(attricorr_test, predcorr_test)))

def test_getweightedsharedinfo(testmodel,
                               test_loader,
                               epoch,
                               prototypefoldername,
                               args):
    testmodel.eval()
    all_digitcaps = torch.zeros((len(test_loader.dataset), testmodel.numcaps, testmodel.out_dim_caps))
    batch_idx = 0
    with torch.no_grad():
        for data in test_loader:
            if args.dataset == "LIDC":
                (x, y_mask, y_attributes, y_mal, _) = data
                x, y_mask, y_attributes, y_mal = x.to("cuda", dtype=torch.float), y_mask.to("cuda", dtype=torch.float), \
                    y_attributes.to("cuda", dtype=torch.float), y_mal.to("cuda",
                                                                         dtype=torch.float)
            else:
                print("only LIDC dataset is supported")
                return

            digitcaps = testmodel.getdigitcaps(x)
            all_digitcaps[batch_idx * args.batch_size:batch_idx * args.batch_size + len(digitcaps)] = digitcaps
            batch_idx += 1
    (std, mean) = torch.std_mean(all_digitcaps, dim=0)
    std, mean = torch.unsqueeze(std, dim=0), torch.unsqueeze(mean, dim=0)
    standardized_digitcaps = torch.div((torch.sub(all_digitcaps, mean)), std)

    # Calculate ytar und yattr mit y=z*w (caps vectors*weights)
    print("Capsule vectors " + str(all_digitcaps.shape))
    y_attr = torch.zeros_like(all_digitcaps)
    attroutlayers = [testmodel.attrOutLayer0, testmodel.attrOutLayer1, testmodel.attrOutLayer2, testmodel.attrOutLayer3,
                     testmodel.attrOutLayer4, testmodel.attrOutLayer5, testmodel.attrOutLayer6, testmodel.attrOutLayer7]
    all_digitcaps_concat = torch.zeros((all_digitcaps.shape[0], all_digitcaps.shape[1] * all_digitcaps.shape[2]))
    for attri in range(all_digitcaps.shape[1]):
        y_attr[:, attri] = torch.mul(all_digitcaps[:, attri, :], attroutlayers[attri][0].weight.cpu())
        all_digitcaps_concat[:, attri * all_digitcaps.shape[-1]:(attri + 1) * all_digitcaps.shape[-1]] = all_digitcaps[
                                                                                                         :, attri, :]
    y_tar = torch.mul(torch.unsqueeze(all_digitcaps_concat, 1),
                      torch.unsqueeze(testmodel.predOutLayers[1].weight.cpu(), dim=0))
    print(y_attr.shape)
    print(y_tar.shape)

    # Calculate w'tar und w'attr mit w'=y/s'
    print("Stand. caps vect: " + str(standardized_digitcaps.shape))
    w_attr_stand = torch.zeros((standardized_digitcaps.shape[1], standardized_digitcaps.shape[2]))
    standardized_digitcaps_concat = torch.zeros(
        (standardized_digitcaps.shape[0], standardized_digitcaps.shape[1] * standardized_digitcaps.shape[2]))
    for attri in range(all_digitcaps.shape[1]):
        w_attr_stand[attri, :] = torch.nanmean(torch.div(y_attr[:, attri], standardized_digitcaps[:, attri]), dim=0)
        standardized_digitcaps_concat[:,
        attri * standardized_digitcaps.shape[-1]:(attri + 1) * standardized_digitcaps.shape[
            -1]] = standardized_digitcaps[:, attri, :]
    print(y_tar.shape)
    print(torch.unsqueeze(standardized_digitcaps_concat, dim=1).shape)
    w_tar_stand = torch.nanmean(torch.div(y_tar, torch.unsqueeze(standardized_digitcaps_concat, dim=1)), dim=0)
    print(w_attr_stand.shape)
    print(w_tar_stand.shape)

    print("WEIGHTS")
    print(testmodel.predOutLayers[1].weight.shape)
    print(testmodel.predOutLayers[1].bias.shape)
    print(testmodel.attrOutLayer0[0].weight.shape)
    print(testmodel.attrOutLayer0[0].bias.shape)
    print("stand. z values, w'=y:z'")
    print("use abs(w')")
    w_tar_stand = torch.abs(w_tar_stand)
    w_attr_stand = torch.abs(w_attr_stand)
    spearman_corr = torch.zeros((8, 5))
    pearson_corr = torch.zeros((8, 5))
    for attri in range(8):
        for malclassi in range(5):
            spearman_corr[attri, malclassi] = torchmetrics.functional.spearman_corrcoef(
                w_tar_stand[malclassi, attri * testmodel.out_dim_caps:(attri + 1) * testmodel.out_dim_caps],
                w_attr_stand[attri])
            pearson_corr[attri, malclassi] = torchmetrics.functional.pearson_corrcoef(
                w_tar_stand[malclassi, attri * testmodel.out_dim_caps:(attri + 1) * testmodel.out_dim_caps],
                w_attr_stand[attri])
    print("pearson_corr_weightsonly mean: " + str(torch.nanmean(torch.abs(pearson_corr))))
    print("spearman_corr_weightsonly mean: " + str(torch.nanmean(torch.abs(spearman_corr))))
    plt.suptitle("mean over n")
    plt.subplot(1, 2, 1)
    plt.title("Absolute Spearman, mearn " + str(torch.nanmean(torch.abs(spearman_corr)).item()))
    plt.imshow(torch.abs(spearman_corr), cmap="RdYlGn")
    plt.ylabel("attributes")
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7], ["sub", "is", "cal", "sph", "mar", "lob", "spic", "tex"])
    plt.xlabel("mal_classes")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.title("Absolute Pearson, mean: " + str(torch.nanmean(torch.abs(pearson_corr)).item()))
    plt.imshow(torch.abs(pearson_corr).detach().numpy(), cmap="RdYlGn")
    plt.ylabel("attributes")
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7], ["sub", "is", "cal", "sph", "mar", "lob", "spic", "tex"])
    plt.xlabel("mal_classes")
    plt.colorbar()
    plt.show()

def test_indepth(testmodel, data_loader, epoch,
                 prototypefoldername, args):
    """
    Test model and use prototype scores for accuracies
    :param testmodel: model to be tested
    :param data_loader: data_loader used for testing
    :param epoch: number of epoch the model is being tested
    :param prototypefoldername: folder direction of prototypes
    :return: [mal_accuracy, attribute_accuracies, dice score]
    """
    print("test in depth")
    testmodel.eval()
    correct_mal = 0
    if args.dataset in ["LIDC", "derm7pt", "FunnyNodules"]:
        usedProtos_allcpsi = []
        for i in range(testmodel.numcaps):
            usedProtos_allcpsi.append(
                torch.zeros((testmodel.protodigis_list[i].shape[0], testmodel.protodigis_list[i].shape[1]), dtype=int))
    elif args.dataset == "Chexbert":
        usedProtos0 = torch.zeros((testmodel.protodigis0.shape[0], testmodel.protodigis0.shape[1]), dtype=int)
        usedProtos1 = torch.zeros((testmodel.protodigis1.shape[0], testmodel.protodigis1.shape[1]), dtype=int)
        usedProtos2 = torch.zeros((testmodel.protodigis2.shape[0], testmodel.protodigis2.shape[1]), dtype=int)
        usedProtos3 = torch.zeros((testmodel.protodigis3.shape[0], testmodel.protodigis3.shape[1]), dtype=int)
        usedProtos4 = torch.zeros((testmodel.protodigis4.shape[0], testmodel.protodigis4.shape[1]), dtype=int)
        usedProtos5 = torch.zeros((testmodel.protodigis5.shape[0], testmodel.protodigis5.shape[1]), dtype=int)
        usedProtos6 = torch.zeros((testmodel.protodigis6.shape[0], testmodel.protodigis6.shape[1]), dtype=int)
        usedProtos7 = torch.zeros((testmodel.protodigis7.shape[0], testmodel.protodigis7.shape[1]), dtype=int)
        usedProtos8 = torch.zeros((testmodel.protodigis8.shape[0], testmodel.protodigis8.shape[1]), dtype=int)
        usedProtos9 = torch.zeros((testmodel.protodigis9.shape[0], testmodel.protodigis9.shape[1]), dtype=int)
        usedProtos10 = torch.zeros((testmodel.protodigis10.shape[0], testmodel.protodigis10.shape[1]), dtype=int)
        usedProtos11 = torch.zeros((testmodel.protodigis11.shape[0], testmodel.protodigis11.shape[1]), dtype=int)
        usedProtos12 = torch.zeros((testmodel.protodigis12.shape[0], testmodel.protodigis12.shape[1]), dtype=int)
        usedProtos_allcpsi = [usedProtos0, usedProtos1, usedProtos2, usedProtos3, usedProtos4, usedProtos5, usedProtos6,
                              usedProtos7, usedProtos8, usedProtos9, usedProtos10, usedProtos11, usedProtos12]

    if args.dataset == "LIDC":
        num_attributes = 8
    elif args.dataset == "derm7pt":
        num_attributes = 7
    elif args.dataset == "FunnyNodules":
        num_attributes = 6
    correct_attproto = torch.zeros((num_attributes,))
    distance_attr_gt = torch.zeros((num_attributes,))
    distance_attr_pred = torch.zeros((num_attributes,))
    howoftenareattrpredandattrprotosame = torch.zeros((num_attributes,))
    if args.dataset == "Chexbert":
        correct_attproto = torch.zeros((13,))
        target_confusionmatrix = torch.zeros((2, 2))
        attris_confusionmatrix = torch.zeros((num_attributes, 2, 2))
        target_auc = 0
        target_auc_gt = []
        target_auc_pred = []
    dc_score = 0
    with torch.no_grad():

        # in depth analysis of prototypes
        r_counter = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
        r_allprotolabels = {0.0: [],
                            0.25: [],
                            0.5: [],
                            0.75: [],
                            1.0: []}
        sp_counter = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
        sp_allprotolabels = {0.0: [],
                            0.25: [],
                            0.5: [],
                            0.75: [],
                            1.0: []}
        es_counter = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
        es_allprotolabels = {0.0: [],
                            0.25: [],
                            0.5: [],
                            0.75: [],
                            1.0: []}
        s_counter = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
        s_allprotolabels = {0.0: [],
                            0.25: [],
                            0.5: [],
                            0.75: [],
                            1.0: []}
        i_counter = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
        i_allprotolabels = {0.0: [],
                            0.25: [],
                            0.5: [],
                            0.75: [],
                            1.0: []}
        is_counter = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
        is_allprotolabels = {0.0: [],
                            1.0: []}

        for i, data in enumerate(data_loader):
            print(str(i) + " / " + str(len(data_loader)))
            if args.dataset == "LIDC":
                (x, y_mask, y_attributes, y_mal, _, _) = data
                x, y_mask, y_attributes, y_mal = x.to("cuda", dtype=torch.float), y_mask.to("cuda",
                                                                                            dtype=torch.float), \
                    y_attributes.to("cuda", dtype=torch.float), y_mal.to("cuda",
                                                                         dtype=torch.float)
            elif args.dataset == "derm7pt":
                (x, y_mask, y_attributes, y_mal, sampleID) = data
                x, y_mask, y_attributes, y_mal = x.to("cuda", dtype=torch.float), y_mask.to("cuda", dtype=torch.float), \
                    y_attributes.to("cuda", dtype=torch.int64), y_mal.to("cuda", dtype=torch.int64)
            elif args.dataset == "Chexbert":
                (x, y_mal, y_attributes, sampleID) = data
                y_mask = torch.tensor(0)
                x, y_mal, y_attributes, y_mask = x.to("cuda", dtype=torch.float), y_mal.to("cuda", dtype=torch.float), \
                    y_attributes.to("cuda", dtype=torch.float), y_mask.to("cuda", dtype=torch.float)
            elif args.dataset == "FunnyNodules":
                (x, y_mask, y_attributes, y_mal, sampleID) = data
                x, y_attributes, y_mal = x.to("cuda", dtype=torch.float), \
                    y_attributes.to("cuda", dtype=torch.float), y_mal.to("cuda", dtype=torch.float)

            if args.base_model == "ViT":
                dists_to_protos = testmodel.getDistance(x)
                pred_outs, pred_mask = testmodel(x)
                if args.dataset == "LIDC":
                    pred_attr = pred_outs[:, :8]
                elif args.dataset == "derm7pt":
                    pred_attr = []
                    pred_attr.append(pred_outs[:, 0:3])
                    pred_attr.append(pred_outs[:, 3:5])
                    pred_attr.append(pred_outs[:, 5:8])
                    pred_attr.append(pred_outs[:, 8:11])
                    pred_attr.append(pred_outs[:, 11:14])
                    pred_attr.append(pred_outs[:, 14:17])
                    pred_attr.append(pred_outs[:, 17:19])
                elif args.dataset == "FunnyNodules":
                    pred_attr = pred_outs[:, :6]
                    pred_mal = pred_outs[:, 6:]
            else:
                _, pred_attr, pred_mask, dists_to_protos = testmodel(x)

            if args.dataset in ["LIDC"]:
                dice = Dice(average='micro').to("cuda")
                if args.threeD:
                    dc_score += (dice(pred_mask.permute(0, 1, 4, 2, 3), y_mask.type(torch.int64)) * y_mask.shape[0])
                else:
                    dc_score += (dice(pred_mask, y_mask.type(torch.int64)) * y_mask.shape[0])

            _, min_proto_idx0 = torch.min(torch.flatten(dists_to_protos[0], start_dim=1), dim=-1)
            min_proto_idx0 = [unravel_index(i, dists_to_protos[0][0].shape) for i in min_proto_idx0]
            _, min_proto_idx1 = torch.min(torch.flatten(dists_to_protos[1], start_dim=1), dim=-1)
            min_proto_idx1 = [unravel_index(i, dists_to_protos[1][0].shape) for i in min_proto_idx1]
            _, min_proto_idx2 = torch.min(torch.flatten(dists_to_protos[2], start_dim=1), dim=-1)
            min_proto_idx2 = [unravel_index(i, dists_to_protos[2][0].shape) for i in min_proto_idx2]
            _, min_proto_idx3 = torch.min(torch.flatten(dists_to_protos[3], start_dim=1), dim=-1)
            min_proto_idx3 = [unravel_index(i, dists_to_protos[3][0].shape) for i in min_proto_idx3]
            _, min_proto_idx4 = torch.min(torch.flatten(dists_to_protos[4], start_dim=1), dim=-1)
            min_proto_idx4 = [unravel_index(i, dists_to_protos[4][0].shape) for i in min_proto_idx4]
            if args.dataset == "FunnyNodules":
                _, min_proto_idx5 = torch.min(torch.flatten(dists_to_protos[5], start_dim=1), dim=-1)
                min_proto_idx5 = [unravel_index(i, dists_to_protos[5][0].shape) for i in min_proto_idx5]
            if args.dataset == "derm7pt":
                _, min_proto_idx5 = torch.min(torch.flatten(dists_to_protos[5], start_dim=1), dim=-1)
                min_proto_idx5 = [unravel_index(i, dists_to_protos[5][0].shape) for i in min_proto_idx5]
                _, min_proto_idx6 = torch.min(torch.flatten(dists_to_protos[6], start_dim=1), dim=-1)
                min_proto_idx6 = [unravel_index(i, dists_to_protos[6][0].shape) for i in min_proto_idx6]
            if args.dataset == "LIDC":
                _, min_proto_idx5 = torch.min(torch.flatten(dists_to_protos[5], start_dim=1), dim=-1)
                min_proto_idx5 = [unravel_index(i, dists_to_protos[5][0].shape) for i in min_proto_idx5]
                _, min_proto_idx6 = torch.min(torch.flatten(dists_to_protos[6], start_dim=1), dim=-1)
                min_proto_idx6 = [unravel_index(i, dists_to_protos[6][0].shape) for i in min_proto_idx6]
                _, min_proto_idx7 = torch.min(torch.flatten(dists_to_protos[7], start_dim=1), dim=-1)
                min_proto_idx7 = [unravel_index(i, dists_to_protos[7][0].shape) for i in min_proto_idx7]
            if args.dataset in ["LIDC", "derm7pt", "FunnyNodules"]:
                min_proto_idx_allcaps = []
                max_proto_idx_allcaps = []
                for j in range(testmodel.numcaps):
                    min_proto_idx_allcaps.append([unravel_index(i, dists_to_protos[j][0].shape) for i in
                                                  torch.min(torch.flatten(dists_to_protos[j], start_dim=1), dim=-1)[1]])
                    max_proto_idx_allcaps.append([unravel_index(i, dists_to_protos[j][0].shape) for i in
                                                  torch.max(torch.flatten(dists_to_protos[j], start_dim=1), dim=-1)[1]])
            elif args.dataset == "Chexbert":
                _, min_proto_idx8 = torch.min(torch.flatten(dists_to_protos[8], start_dim=1), dim=-1)
                min_proto_idx8 = [unravel_index(i, dists_to_protos[8][0].shape) for i in min_proto_idx8]
                _, min_proto_idx9 = torch.min(torch.flatten(dists_to_protos[9], start_dim=1), dim=-1)
                min_proto_idx9 = [unravel_index(i, dists_to_protos[9][0].shape) for i in min_proto_idx9]
                _, min_proto_idx10 = torch.min(torch.flatten(dists_to_protos[10], start_dim=1), dim=-1)
                min_proto_idx10 = [unravel_index(i, dists_to_protos[10][0].shape) for i in min_proto_idx10]
                _, min_proto_idx11 = torch.min(torch.flatten(dists_to_protos[11], start_dim=1), dim=-1)
                min_proto_idx11 = [unravel_index(i, dists_to_protos[11][0].shape) for i in min_proto_idx11]
                _, min_proto_idx12 = torch.min(torch.flatten(dists_to_protos[12], start_dim=1), dim=-1)
                min_proto_idx12 = [unravel_index(i, dists_to_protos[12][0].shape) for i in min_proto_idx12]
                min_proto_idx_allcaps = [min_proto_idx0, min_proto_idx1, min_proto_idx2, min_proto_idx3,
                                         min_proto_idx4, min_proto_idx5, min_proto_idx6, min_proto_idx7,
                                         min_proto_idx8, min_proto_idx9, min_proto_idx10, min_proto_idx11,
                                         min_proto_idx12]
            if args.base_model == "ViT":
                x_ex = torch.zeros((x.shape[0], len(min_proto_idx_allcaps), testmodel.protodigis0.shape[-2], testmodel.protodigis0.shape[-1]))
            else:
                x_ex = torch.zeros((x.shape[0], len(min_proto_idx_allcaps), testmodel.protodigis0.shape[-1]))
            attrLabelsSamples_protos = torch.zeros_like(y_attributes)
            attrLabelsSamples_protos_max = torch.zeros_like(y_attributes)
            folder_dir = "./prototypes/" + str(prototypefoldername) + "/" + str(epoch)
            for sai in range(x.shape[0]):
                for capsule_idx in range(len(min_proto_idx_allcaps)):
                    x_ex[sai, capsule_idx, :] = testmodel.protodigis_list[capsule_idx][
                        min_proto_idx_allcaps[capsule_idx][sai][0], min_proto_idx_allcaps[capsule_idx][sai][1]]
                    usedProtos_allcpsi[capsule_idx][
                        min_proto_idx_allcaps[capsule_idx][sai][0], min_proto_idx_allcaps[capsule_idx][sai][1]] += 1
                    for image_name in os.listdir(folder_dir):
                        if image_name.startswith(
                                "cpslnr" + str(capsule_idx) + "_protonr" + str(
                                    min_proto_idx_allcaps[capsule_idx][sai][0]) + "-" + str(
                                    min_proto_idx_allcaps[capsule_idx][sai][1]) + "_"):
                            proto_scores_str = image_name.split("[")[1].split("]")[0].split(", ")
                            proto_scores = torch.tensor(np.array(proto_scores_str).astype(np.float64))
                            attrLabelsSamples_protos[sai, capsule_idx] = proto_scores[capsule_idx]
                            # print("in test in depth")
                            # print(proto_scores) # [attributes,target]
                            if args.dataset =="FunnyNodules" and capsule_idx==0:
                                # print(int(proto_scores[capsule_idx]*4))
                                r_counter[int(proto_scores[capsule_idx]*4)][min_proto_idx_allcaps[capsule_idx][sai][1]] += 1
                                r_allprotolabels[proto_scores[capsule_idx].item()].append(proto_scores)
                            if args.dataset =="FunnyNodules" and capsule_idx==1:
                                # print(int(proto_scores[capsule_idx]*4))
                                sp_counter[int(proto_scores[capsule_idx]*4)][min_proto_idx_allcaps[capsule_idx][sai][1]] += 1
                                sp_allprotolabels[proto_scores[capsule_idx].item()].append(proto_scores)
                            if args.dataset =="FunnyNodules" and capsule_idx==2:
                                # print(int(proto_scores[capsule_idx]*4))
                                es_counter[int(proto_scores[capsule_idx]*4)][min_proto_idx_allcaps[capsule_idx][sai][1]] += 1
                                es_allprotolabels[proto_scores[capsule_idx].item()].append(proto_scores)
                            if args.dataset =="FunnyNodules" and capsule_idx==3:
                                # print(int(proto_scores[capsule_idx]*4))
                                s_counter[int(proto_scores[capsule_idx]*4)][min_proto_idx_allcaps[capsule_idx][sai][1]] += 1
                                s_allprotolabels[proto_scores[capsule_idx].item()].append(proto_scores)
                            if args.dataset =="FunnyNodules" and capsule_idx==4:
                                # print(int(proto_scores[capsule_idx]*4))
                                i_counter[int(proto_scores[capsule_idx]*4)][min_proto_idx_allcaps[capsule_idx][sai][1]] += 1
                                i_allprotolabels[proto_scores[capsule_idx].item()].append(proto_scores)
                            if args.dataset =="FunnyNodules" and capsule_idx==5:
                                # print(int(proto_scores[capsule_idx]*4))
                                is_counter[int(proto_scores[capsule_idx])][min_proto_idx_allcaps[capsule_idx][sai][1]] += 1
                                is_allprotolabels[proto_scores[capsule_idx].item()].append(proto_scores)

                        if image_name.startswith(
                                "cpslnr" + str(capsule_idx) + "_protonr" + str(
                                    max_proto_idx_allcaps[capsule_idx][sai][0]) + "-" + str(
                                    max_proto_idx_allcaps[capsule_idx][sai][1]) + "_"):
                            proto_scores_str = image_name.split("[")[1].split("]")[0].split(", ")
                            proto_scores = torch.tensor(np.array(proto_scores_str).astype(np.float64))
                            attrLabelsSamples_protos_max[sai, capsule_idx] = proto_scores[capsule_idx]

            if args.base_model == "ViT":
                pred_p = testmodel.forwardprotodigis(x_ex=x_ex.to("cuda", dtype=torch.float))
            else:
                pred_p, _, _ = testmodel.forwardCapsule(x_ex=x_ex.to("cuda", dtype=torch.float))

            if args.dataset in ["LIDC","derm7pt", "FunnyNodules"]:
                if args.base_model == "ViT":
                    if args.dataset == "derm7pt":
                        mal_confusion_matrix = confusion_matrix(y_mal.cpu().detach().numpy(),
                                                                torch.argmax(pred_p,
                                                                             dim=-1).cpu().detach().numpy(),
                                                                labels=[0, 1, 2, 3, 4])
                    elif args.dataset == "LIDC":
                        pred_p *= 4
                        pred_p = torch.round(pred_p)
                        mal_confusion_matrix = confusion_matrix(np.argmax(y_mal.cpu().detach().numpy(), axis=1) + 1,
                                                                pred_p.cpu().detach().numpy() + 1,
                                                                labels=[1, 2, 3, 4, 5])
                    elif args.dataset == "FunnyNodules":
                        pred_p *= 4
                        pred_p = torch.round(pred_p)
                        y_mal *= 4
                        mal_confusion_matrix = confusion_matrix(y_mal.cpu().detach().numpy() + 1,
                                                                torch.squeeze(pred_p).cpu().detach().numpy() + 1,
                                                                labels=[1, 2, 3, 4, 5])

                elif args.dataset == "derm7pt":
                    mal_confusion_matrix = confusion_matrix(y_mal.cpu().detach().numpy(),
                                                            torch.argmax(pred_p, dim=-1).cpu().detach().numpy(),
                                                            labels=[0, 1, 2, 3, 4])
                elif args.dataset == "FunnyNodules":
                    pred_p *= 4
                    pred_p = torch.round(pred_p)
                    y_mal *= 4
                    mal_confusion_matrix = confusion_matrix(y_mal.cpu().detach().numpy() + 1,
                                                            torch.squeeze(pred_p).cpu().detach().numpy() + 1,
                                                            labels=[1, 2, 3, 4, 5])
                else:
                    mal_confusion_matrix = confusion_matrix(np.argmax(y_mal.cpu().detach().numpy(), axis=1) + 1,
                                                            np.argmax(pred_p.cpu().detach().numpy(), axis=1) + 1,
                                                            labels=[1, 2, 3, 4, 5])

                if args.dataset == "derm7pt":
                    mal_correct_within_one = sum(np.diagonal(mal_confusion_matrix, offset=0))
                    print("confusion matrix target")
                    print(mal_confusion_matrix)
                elif args.dataset == "LIDC":
                    mal_correct_within_one = sum(np.diagonal(mal_confusion_matrix, offset=0)) + \
                                             sum(np.diagonal(mal_confusion_matrix, offset=1)) + \
                                             sum(np.diagonal(mal_confusion_matrix, offset=-1))
                elif args.dataset == "FunnyNodules":
                    mal_correct_within_one = sum(np.diagonal(mal_confusion_matrix, offset=0)) + \
                                             sum(np.diagonal(mal_confusion_matrix, offset=1)) + \
                                             sum(np.diagonal(mal_confusion_matrix, offset=-1))
                correct_mal += mal_correct_within_one


                if args.dataset == "derm7pt":
                    print(y_attributes.shape)
                    print(attrLabelsSamples_protos_max.shape)
                    for attri_i in range(7):
                        correct_attproto[attri_i] += torch.sum(y_attributes[:,attri_i] == attrLabelsSamples_protos[:,attri_i]).cpu().detach().numpy()
                        distance_attr_gt[attri_i] += y_attributes.shape[0] - torch.sum(y_attributes[:,attri_i] == attrLabelsSamples_protos_max[:,attri_i]).cpu().detach().numpy()
                        howoftenareattrpredandattrprotosame[attri_i] += torch.sum(attrLabelsSamples_protos[:,attri_i] == torch.argmax(pred_attr[attri_i],dim=-1)).cpu().detach().numpy()

                elif args.dataset == "LIDC":
                    y_attributes[:, [0, 3, 4, 5, 6, 7]] *= 4
                    y_attributes[:, 1] *= 3
                    y_attributes[:, 2] *= 5
                    y_attributes += 1
                    pred_attr[:, [0, 3, 4, 5, 6, 7]] *= 4
                    pred_attr[:, 1] *= 3
                    pred_attr[:, 2] *= 5
                    pred_attr += 1
                    attrLabelsSamples_protos[:, [0, 3, 4, 5, 6, 7]] *= 4
                    attrLabelsSamples_protos[:, 1] *= 3
                    attrLabelsSamples_protos[:, 2] *= 5
                    attrLabelsSamples_protos += 1
                    attrLabelsSamples_protos_max[:, [0, 3, 4, 5, 6, 7]] *= 4
                    attrLabelsSamples_protos_max[:, 1] *= 3
                    attrLabelsSamples_protos_max[:, 2] *= 5
                    attrLabelsSamples_protos_max += 1
                    for at in range(y_attributes.shape[1]):
                        a_labels = [1, 2, 3, 4, 5]
                        if num_attributes == 8:
                            if at == 1:
                                a_labels = [1, 2, 3, 4]
                            if at == 2:
                                a_labels = [1, 2, 3, 4, 5, 6]
                        attr_confusion_matrix_proto = confusion_matrix(
                            np.rint(y_attributes[:, at].cpu().detach().numpy()),
                            np.rint(attrLabelsSamples_protos[:, at].cpu().detach().numpy()),
                            labels=a_labels)
                        correct_attproto[at] += sum(np.diagonal(attr_confusion_matrix_proto, offset=0)) + \
                                                sum(np.diagonal(attr_confusion_matrix_proto, offset=1)) + \
                                                sum(np.diagonal(attr_confusion_matrix_proto, offset=-1))
                        howoftenareattrpredandattrprotosame[at] += np.sum(np.absolute((np.rint(attrLabelsSamples_protos[:, at].cpu().detach().numpy())
                                                                                        -np.rint(pred_attr[:,at].cpu().detach().numpy())))<2)
                        attr_confusion_matrix_proto_max_gt = confusion_matrix(
                            np.rint(y_attributes[:, at].cpu().detach().numpy()),
                            np.rint(attrLabelsSamples_protos_max[:, at].cpu().detach().numpy()),
                            labels=a_labels)
                        distance_attr_gt[at] += attr_confusion_matrix_proto_max_gt.sum() - \
                                             (sum(np.diagonal(attr_confusion_matrix_proto_max_gt, offset=0)) + \
                                             sum(np.diagonal(attr_confusion_matrix_proto_max_gt, offset=1)) + \
                                             sum(np.diagonal(attr_confusion_matrix_proto_max_gt, offset=-1)))
                        attr_confusion_matrix_proto_max_pred = confusion_matrix(
                            np.rint(attrLabelsSamples_protos[:, at].cpu().detach().numpy()),
                            np.rint(attrLabelsSamples_protos_max[:, at].cpu().detach().numpy()),
                            labels=a_labels)
                        distance_attr_pred[at] += attr_confusion_matrix_proto_max_pred.sum() - \
                                             (sum(np.diagonal(attr_confusion_matrix_proto_max_pred, offset=0)) + \
                                             sum(np.diagonal(attr_confusion_matrix_proto_max_pred, offset=1)) + \
                                             sum(np.diagonal(attr_confusion_matrix_proto_max_pred, offset=-1)))

                elif args.dataset == "FunnyNodules":
                    y_attributes[:, :-1] *= 4
                    y_attributes[:, :-1] += 1
                    pred_attr[:, :-1] *= 4
                    pred_attr[:, :-1] += 1
                    attrLabelsSamples_protos[:, :-1] *= 4
                    attrLabelsSamples_protos[:, :-1] += 1
                    attrLabelsSamples_protos_max[:, :-1] *= 4
                    attrLabelsSamples_protos_max[:, :-1] += 1
                    for at in range(y_attributes.shape[1]):
                        if at < 5:
                            a_labels = [1, 2, 3, 4, 5]
                        else:
                            a_labels = [0,1]
                        attr_confusion_matrix_proto = confusion_matrix(
                            np.rint(y_attributes[:, at].cpu().detach().numpy()).astype(int),
                            np.rint(attrLabelsSamples_protos[:, at].cpu().detach().numpy()).astype(int),
                            labels=a_labels)
                        if at < 5:
                            correct_attproto[at] += sum(np.diagonal(attr_confusion_matrix_proto, offset=0)) + \
                                                    sum(np.diagonal(attr_confusion_matrix_proto, offset=1)) + \
                                                    sum(np.diagonal(attr_confusion_matrix_proto, offset=-1))
                            howoftenareattrpredandattrprotosame[at] += np.sum(np.absolute((np.rint(attrLabelsSamples_protos[:, at].cpu().detach().numpy()).astype(int)
                                                                                        -np.rint(pred_attr[:,at].cpu().detach().numpy()).astype(int)))<2)
                        else:
                            correct_attproto[at] += sum(np.diagonal(attr_confusion_matrix_proto, offset=0))
                            howoftenareattrpredandattrprotosame[at] += np.sum(
                                (np.rint(attrLabelsSamples_protos[:, at].cpu().detach().numpy()).astype(int)
                                             == np.rint(pred_attr[:, at].cpu().detach().numpy()).astype(int)))
                        attr_confusion_matrix_proto_max_gt = confusion_matrix(
                            np.rint(y_attributes[:, at].cpu().detach().numpy()).astype(int),
                            np.rint(attrLabelsSamples_protos_max[:, at].cpu().detach().numpy()).astype(int),
                            labels=a_labels)
                        if at < 5:
                            distance_attr_gt[at] += attr_confusion_matrix_proto_max_gt.sum() - \
                                                 (sum(np.diagonal(attr_confusion_matrix_proto_max_gt, offset=0)) + \
                                                 sum(np.diagonal(attr_confusion_matrix_proto_max_gt, offset=1)) + \
                                                 sum(np.diagonal(attr_confusion_matrix_proto_max_gt, offset=-1)))
                        else:
                            distance_attr_gt[at] += attr_confusion_matrix_proto_max_gt.sum() - \
                                                    sum(np.diagonal(attr_confusion_matrix_proto_max_gt, offset=0))
                        attr_confusion_matrix_proto_max_pred = confusion_matrix(
                            np.rint(attrLabelsSamples_protos[:, at].cpu().detach().numpy()).astype(int),
                            np.rint(attrLabelsSamples_protos_max[:, at].cpu().detach().numpy()).astype(int),
                            labels=a_labels)
                        if at < 5:
                            distance_attr_pred[at] += attr_confusion_matrix_proto_max_pred.sum() - \
                                                 (sum(np.diagonal(attr_confusion_matrix_proto_max_pred, offset=0)) + \
                                                 sum(np.diagonal(attr_confusion_matrix_proto_max_pred, offset=1)) + \
                                                 sum(np.diagonal(attr_confusion_matrix_proto_max_pred, offset=-1)))
                        else:
                            distance_attr_pred[at] += attr_confusion_matrix_proto_max_pred.sum() - \
                                                      (sum(np.diagonal(attr_confusion_matrix_proto_max_pred,
                                                                       offset=0)))

            elif args.dataset == "Chexbert":
                target_auc_gt.extend(y_mal.cpu().detach().numpy().tolist())
                target_auc_pred.extend(torch.squeeze(pred_p).cpu().detach().numpy().tolist())
                pred_confusion_matrix = confusion_matrix(
                    y_mal.cpu().detach().numpy(),
                    np.rint(pred_p.cpu().detach().numpy()),
                    labels=[0, 1])
                target_confusionmatrix += pred_confusion_matrix
                mal_correct_within_one = sum(np.diagonal(pred_confusion_matrix, offset=0))
                correct_mal += mal_correct_within_one

                for at in range(y_attributes.shape[1]):
                    attr_confusion_matrix = confusion_matrix(
                        np.rint(y_attributes[:, at].cpu().detach().numpy()),
                        np.rint(attrLabelsSamples_protos[:, at].cpu().detach().numpy()),
                        labels=[0, 1])
                    attris_confusionmatrix[at] += attr_confusion_matrix
                    correct_attproto[at] += sum(np.diagonal(attr_confusion_matrix, offset=0))

    print("roundness proto analysis")
    print(r_counter)
    # print(r_allprotolabels)
    freq_summary = {}
    attribute_names = ["roundness", "spiculation", "edge_sharpness", "size_attr", "intensity", "internal_structure","target"]
    from collections import Counter
    for key, tensor_list in r_allprotolabels.items():
        if not tensor_list:
            print(f"\n=== Key {key} ===\n(empty)")
            continue
        num_indices = tensor_list[0].numel()
        index_counts = [Counter() for _ in range(num_indices)]
        for t in tensor_list:
            for i, val in enumerate(t.tolist()):
                index_counts[i][val] += 1
        freq_summary[key] = index_counts
    for key, index_counts in freq_summary.items():
        print(f"\n=== Key {key} ===")
        for name, counter in zip(attribute_names, index_counts):
            print(f" {name:20s}: {dict(counter)}")

    print("spiculation proto analysis")
    print(sp_counter)
    # print(sp_allprotolabels)
    freq_summary = {}
    attribute_names = ["roundness", "spiculation", "edge_sharpness", "size_attr", "intensity", "internal_structure","target"]
    from collections import Counter
    for key, tensor_list in sp_allprotolabels.items():
        if not tensor_list:
            print(f"\n=== Key {key} ===\n(empty)")
            continue
        num_indices = tensor_list[0].numel()
        index_counts = [Counter() for _ in range(num_indices)]
        for t in tensor_list:
            for i, val in enumerate(t.tolist()):
                index_counts[i][val] += 1
        freq_summary[key] = index_counts
    for key, index_counts in freq_summary.items():
        print(f"\n=== Key {key} ===")
        for name, counter in zip(attribute_names, index_counts):
            print(f" {name:20s}: {dict(counter)}")

    print("edge sharpness proto analysis")
    print(es_counter)
    # print(es_allprotolabels)
    freq_summary = {}
    attribute_names = ["roundness", "spiculation", "edge_sharpness", "size_attr", "intensity", "internal_structure","target"]
    from collections import Counter
    for key, tensor_list in es_allprotolabels.items():
        if not tensor_list:
            print(f"\n=== Key {key} ===\n(empty)")
            continue
        num_indices = tensor_list[0].numel()
        index_counts = [Counter() for _ in range(num_indices)]
        for t in tensor_list:
            for i, val in enumerate(t.tolist()):
                index_counts[i][val] += 1
        freq_summary[key] = index_counts
    for key, index_counts in freq_summary.items():
        print(f"\n=== Key {key} ===")
        for name, counter in zip(attribute_names, index_counts):
            print(f" {name:20s}: {dict(counter)}")

    print("size proto analysis")
    print(s_counter)
    # print(s_allprotolabels)
    freq_summary = {}
    attribute_names = ["roundness", "spiculation", "edge_sharpness", "size_attr", "intensity", "internal_structure","target"]
    from collections import Counter
    for key, tensor_list in s_allprotolabels.items():
        if not tensor_list:
            print(f"\n=== Key {key} ===\n(empty)")
            continue
        num_indices = tensor_list[0].numel()
        index_counts = [Counter() for _ in range(num_indices)]
        for t in tensor_list:
            for i, val in enumerate(t.tolist()):
                index_counts[i][val] += 1
        freq_summary[key] = index_counts
    for key, index_counts in freq_summary.items():
        print(f"\n=== Key {key} ===")
        for name, counter in zip(attribute_names, index_counts):
            print(f" {name:20s}: {dict(counter)}")

    print("intensity proto analysis")
    print(i_counter)
    # print(i_allprotolabels)
    freq_summary = {}
    attribute_names = ["roundness", "spiculation", "edge_sharpness", "size_attr", "intensity", "internal_structure","target"]
    from collections import Counter
    for key, tensor_list in i_allprotolabels.items():
        if not tensor_list:
            print(f"\n=== Key {key} ===\n(empty)")
            continue
        num_indices = tensor_list[0].numel()
        index_counts = [Counter() for _ in range(num_indices)]
        for t in tensor_list:
            for i, val in enumerate(t.tolist()):
                index_counts[i][val] += 1
        freq_summary[key] = index_counts
    for key, index_counts in freq_summary.items():
        print(f"\n=== Key {key} ===")
        for name, counter in zip(attribute_names, index_counts):
            print(f" {name:20s}: {dict(counter)}")

    print("intensity proto analysis")
    print(is_counter)
    # print(is_allprotolabels)
    freq_summary = {}
    attribute_names = ["roundness", "spiculation", "edge_sharpness", "size_attr", "intensity", "internal_structure","target"]
    from collections import Counter
    for key, tensor_list in is_allprotolabels.items():
        if not tensor_list:
            print(f"\n=== Key {key} ===\n(empty)")
            continue
        num_indices = tensor_list[0].numel()
        index_counts = [Counter() for _ in range(num_indices)]
        for t in tensor_list:
            for i, val in enumerate(t.tolist()):
                index_counts[i][val] += 1
        freq_summary[key] = index_counts
    for key, index_counts in freq_summary.items():
        print(f"\n=== Key {key} ===")
        for name, counter in zip(attribute_names, index_counts):
            print(f" {name:20s}: {dict(counter)}")


    if args.dataset == "Chexbert":
        from sklearn import metrics
        fpr, tpr, thresholds = metrics.roc_curve(target_auc_gt, target_auc_pred)
        print(metrics.auc(fpr, tpr))
        target_auc = metrics.auc(fpr, tpr)
    torch.set_printoptions(profile="full")
    print("Prototypes: how many times were prototypes used:")
    print(len(usedProtos_allcpsi))
    print(usedProtos_allcpsi[0].shape)
    print(usedProtos_allcpsi)
    totalnumberofprotos = 0
    usedprotos = 0
    for usedProtoi in range(len(usedProtos_allcpsi)):
        print(torch.count_nonzero(usedProtos_allcpsi[usedProtoi], dim=1))
        totalnumberofprotos += (len(torch.count_nonzero(usedProtos_allcpsi[usedProtoi], dim=1)) * testmodel.numProtos)
        usedprotos += torch.sum(torch.count_nonzero(usedProtos_allcpsi[usedProtoi], dim=1))
    torch.set_printoptions(profile="default")
    print(totalnumberofprotos)
    print("relative used protos " + str(usedprotos / totalnumberofprotos))

    if not args.base_model == "ViT" and args.dataset == "LIDC":
        print("joint information")
        all_attroutLayerweights = [testmodel.attrOutLayer0[0].weight, testmodel.attrOutLayer1[0].weight,
                                   testmodel.attrOutLayer2[0].weight, testmodel.attrOutLayer3[0].weight,
                                   testmodel.attrOutLayer4[0].weight, testmodel.attrOutLayer5[0].weight,
                                   testmodel.attrOutLayer6[0].weight, testmodel.attrOutLayer7[0].weight]
        joint_info = torch.zeros((testmodel.predOutLayers[1].weight.shape[0], len(all_attroutLayerweights)))
        print(joint_info.shape)
        for mal_pred in range(testmodel.predOutLayers[1].weight.shape[0]):
            for cpsi in range(len(all_attroutLayerweights)):
                tobenormed_pred = (testmodel.predOutLayers[1].weight[mal_pred,
                                   cpsi * testmodel.out_dim_caps:(cpsi + 1) * testmodel.out_dim_caps])
                tobenormed_attri = all_attroutLayerweights[cpsi][0]
                count_same_impact_neg_pos = 0
                for cpsdimi in range(testmodel.out_dim_caps):
                    if tobenormed_pred[cpsdimi] * tobenormed_attri[cpsdimi] > 0:
                        count_same_impact_neg_pos += 1
                joint_info[mal_pred, cpsi] = count_same_impact_neg_pos / testmodel.out_dim_caps
        print("joint_info")
        print(joint_info)
        print(torch.mean(joint_info))
        print("benign")
        print(torch.mean(joint_info[0:2, :], dim=0))
        print("malignant")
        print(torch.mean(joint_info[3:, :], dim=0))

    if args.dataset =="LIDC":
        print("distance_attr_gt:")
        print(distance_attr_gt[0] / len(data_loader.dataset))
        print(distance_attr_gt[1] / len(data_loader.dataset))
        print(distance_attr_gt[2] / len(data_loader.dataset))
        print(distance_attr_gt[3] / len(data_loader.dataset))
        print(distance_attr_gt[4] / len(data_loader.dataset))
        print(distance_attr_gt[5] / len(data_loader.dataset))
        print(distance_attr_gt[6] / len(data_loader.dataset))
        print(distance_attr_gt[7] / len(data_loader.dataset))
        print("distance_attr_pred:")
        print(distance_attr_pred[0] / len(data_loader.dataset))
        print(distance_attr_pred[1] / len(data_loader.dataset))
        print(distance_attr_pred[2] / len(data_loader.dataset))
        print(distance_attr_pred[3] / len(data_loader.dataset))
        print(distance_attr_pred[4] / len(data_loader.dataset))
        print(distance_attr_pred[5] / len(data_loader.dataset))
        print(distance_attr_pred[6] / len(data_loader.dataset))
        print(distance_attr_pred[7] / len(data_loader.dataset))
        print("howoftenareattrpredandattrprotosame")
        print(howoftenareattrpredandattrprotosame[0]/ len(data_loader.dataset))
        print(howoftenareattrpredandattrprotosame[1]/ len(data_loader.dataset))
        print(howoftenareattrpredandattrprotosame[2]/ len(data_loader.dataset))
        print(howoftenareattrpredandattrprotosame[3]/ len(data_loader.dataset))
        print(howoftenareattrpredandattrprotosame[4]/ len(data_loader.dataset))
        print(howoftenareattrpredandattrprotosame[5]/ len(data_loader.dataset))
        print(howoftenareattrpredandattrprotosame[6]/ len(data_loader.dataset))
        print(howoftenareattrpredandattrprotosame[7]/ len(data_loader.dataset))
        return correct_mal / len(data_loader.dataset), \
            [correct_attproto[0] / len(data_loader.dataset),
             correct_attproto[1] / len(data_loader.dataset),
             correct_attproto[2] / len(data_loader.dataset),
             correct_attproto[3] / len(data_loader.dataset),
             correct_attproto[4] / len(data_loader.dataset),
             correct_attproto[5] / len(data_loader.dataset),
             correct_attproto[6] / len(data_loader.dataset),
             correct_attproto[7] / len(data_loader.dataset)], \
               dc_score / len(data_loader.dataset)
    elif args.dataset == "derm7pt":
        print("distance_attr_gt:")
        print(distance_attr_gt[0] / len(data_loader.dataset))
        print(distance_attr_gt[1] / len(data_loader.dataset))
        print(distance_attr_gt[2] / len(data_loader.dataset))
        print(distance_attr_gt[3] / len(data_loader.dataset))
        print(distance_attr_gt[4] / len(data_loader.dataset))
        print(distance_attr_gt[5] / len(data_loader.dataset))
        print(distance_attr_gt[6] / len(data_loader.dataset))
        print("howoftenareattrpredandattrprotosame")
        print(howoftenareattrpredandattrprotosame[0]/ len(data_loader.dataset))
        print(howoftenareattrpredandattrprotosame[1]/ len(data_loader.dataset))
        print(howoftenareattrpredandattrprotosame[2]/ len(data_loader.dataset))
        print(howoftenareattrpredandattrprotosame[3]/ len(data_loader.dataset))
        print(howoftenareattrpredandattrprotosame[4]/ len(data_loader.dataset))
        print(howoftenareattrpredandattrprotosame[5]/ len(data_loader.dataset))
        print(howoftenareattrpredandattrprotosame[6]/ len(data_loader.dataset))
        return correct_mal / len(data_loader.dataset), \
            [correct_attproto[0] / len(data_loader.dataset),
             correct_attproto[1] / len(data_loader.dataset),
             correct_attproto[2] / len(data_loader.dataset),
             correct_attproto[3] / len(data_loader.dataset),
             correct_attproto[4] / len(data_loader.dataset),
             correct_attproto[5] / len(data_loader.dataset),
             correct_attproto[6] / len(data_loader.dataset)]
    elif args.dataset == "FunnyNodules":
        print("distance_attr_gt:")
        print(distance_attr_gt[0] / len(data_loader.dataset))
        print(distance_attr_gt[1] / len(data_loader.dataset))
        print(distance_attr_gt[2] / len(data_loader.dataset))
        print(distance_attr_gt[3] / len(data_loader.dataset))
        print(distance_attr_gt[4] / len(data_loader.dataset))
        print(distance_attr_gt[5] / len(data_loader.dataset))
        print("howoftenareattrpredandattrprotosame")
        print(howoftenareattrpredandattrprotosame[0]/ len(data_loader.dataset))
        print(howoftenareattrpredandattrprotosame[1]/ len(data_loader.dataset))
        print(howoftenareattrpredandattrprotosame[2]/ len(data_loader.dataset))
        print(howoftenareattrpredandattrprotosame[3]/ len(data_loader.dataset))
        print(howoftenareattrpredandattrprotosame[4]/ len(data_loader.dataset))
        print(howoftenareattrpredandattrprotosame[5]/ len(data_loader.dataset))
        return correct_mal / len(data_loader.dataset), \
            [correct_attproto[0] / len(data_loader.dataset),
             correct_attproto[1] / len(data_loader.dataset),
             correct_attproto[2] / len(data_loader.dataset),
             correct_attproto[3] / len(data_loader.dataset),
             correct_attproto[4] / len(data_loader.dataset),
             correct_attproto[5] / len(data_loader.dataset)]
    elif args.dataset == "Chexbert":
        target_accuracy = sum(np.diagonal(target_confusionmatrix, offset=0)) / len(data_loader.dataset)
        target_precision = target_confusionmatrix[1, 1] / (target_confusionmatrix[1, 1] + target_confusionmatrix[1, 0])
        target_recall = target_confusionmatrix[1, 1] / (target_confusionmatrix[1, 1] + target_confusionmatrix[0, 1])
        target_f1 = 2 * ((target_precision * target_recall) / (target_precision + target_recall))
        return [target_auc, target_accuracy, target_precision, target_recall, target_f1], \
            [correct_attproto[0] / len(data_loader.dataset),
             correct_attproto[1] / len(data_loader.dataset),
             correct_attproto[2] / len(data_loader.dataset),
             correct_attproto[3] / len(data_loader.dataset),
             correct_attproto[4] / len(data_loader.dataset),
             correct_attproto[5] / len(data_loader.dataset),
             correct_attproto[6] / len(data_loader.dataset),
             correct_attproto[7] / len(data_loader.dataset),
             correct_attproto[8] / len(data_loader.dataset),
             correct_attproto[9] / len(data_loader.dataset),
             correct_attproto[10] / len(data_loader.dataset),
             correct_attproto[11] / len(data_loader.dataset),
             correct_attproto[12] / len(data_loader.dataset)]


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append((index % dim).item())
        index = index // dim
    return tuple(reversed(out))


# ----- FOR FUNNYNODULES ANALYSIS -----
def test_contrastivity(model, args):
    import pandas as pd
    attris = ["roundness", "spiculation", "edge_sharpness", "size_attr", "intensity", "internal_structure"]
    results = []
    for s in list(range(0, 100)):
        for attris_idx, attr_name in enumerate(attris):
            dl = create_attribute_dataloader(attr_name, base_seed=s)
            allimgs, allattrs, alltars = [], [], []
            for imgs, masks, attrs, targets, idxs in dl:
                for i in range(len(imgs)):
                    allimgs.append(imgs[i, 0])
                    allattrs.append(attrs[i])
                    alltars.append(targets[i])
            num_variations = 5 if attr_name != "internal_structure" else 2
            # fig, axs = plt.subplots(1,num_variations)
            # fig.suptitle(attr_name,fontsize=12)
            # for i in range(num_variations):
            #     axs[i].imshow(allimgs[i].cpu().numpy(), cmap="gray")
            #     axs[i].set_title("tar:"+str(alltars[i].item())+"a:"+str(allattrs[i][attris_idx].item()),fontsize=12)
            #     axs[i].axis("off")
            # plt.show()
            _, test_acc_contra, te_attr_acc_contra = test(testmodel=model, data_loader=dl, args=args)
            entry = {
                "seed": s,
                "varied_attribute": attr_name,
                "test_acc_contra": float(test_acc_contra),
            }
            for i, attr in enumerate(attris):
                entry[f"attr_acc_{attr}"] = float(te_attr_acc_contra[i].item())
            results.append(entry)
    results_df = pd.DataFrame(results)
    summary = results_df.groupby("varied_attribute")[
        ["test_acc_contra",
         "attr_acc_roundness", "attr_acc_spiculation", "attr_acc_edge_sharpness",
         "attr_acc_size_attr", "attr_acc_intensity", "attr_acc_internal_structure"]
    ].agg(["mean", "std"])
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 0)
    pd.set_option('display.max_colwidth', None)
    print(summary.round(3))
    # === Plot model test accuracy by varied attribute ===
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # Left plot: test accuracy per varied attribute
    results_df.boxplot(column="test_acc_contra", by="varied_attribute", ax=axs[0], fontsize=12)
    axs[0].set_title("Model Test Accuracy by Varied Attribute",fontsize=12)
    axs[0].set_xlabel("Varied Attribute",fontsize=12)
    axs[0].set_ylabel("Accuracy",fontsize=12)
    # Right plot: heatmap-style barplot for attribute accuracies
    attr_cols = [
        "attr_acc_roundness", "attr_acc_spiculation", "attr_acc_edge_sharpness",
        "attr_acc_size_attr", "attr_acc_intensity", "attr_acc_internal_structure"
    ]
    # Compute mean accuracies for each varied_attribute
    attr_summary = results_df.groupby("varied_attribute")[attr_cols].mean()
    attr_summary.plot(kind="bar", ax=axs[1],fontsize=12)
    axs[1].set_title("Attribute Prediction Accuracies (mean per varied attribute)",fontsize=12)
    axs[1].set_xlabel("Varied Attribute",fontsize=12)
    axs[1].set_ylabel("Accuracy",fontsize=12)
    axs[1].legend(title="Predicted Attribute", bbox_to_anchor=(1.05, 1), loc="upper left",fontsize=12)
    plt.tight_layout()
    plt.show()

def visualize_all_attribute_responses(model, args,):
    attributes = [
        "roundness", "spiculation", "edge_sharpness",
        "size_attr", "intensity", "internal_structure"]
    seeds =  list(range(0, 100))
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
                    if args.base_model == "ViT":
                        pred_outs, _ = model(imgs)
                        if args.dataset == "FunnyNodules":
                            preds = pred_outs[:, 6:]
                    elif args.base_model in ["ResNetMT","DenseNetMT"]:
                        imgs = imgs.repeat(1, 3, 1, 1)
                        outputs = model(imgs)
                        preds = outputs[:, 0]
                    else:
                        if args.dataset == "FunnyNodules":
                            preds, _, _, _ = model(imgs)
                    preds = preds.detach().cpu()
                    targets = targets.detach().cpu()

                    pred_targets.extend(preds.numpy().tolist())
                    true_targets.extend(targets.numpy().tolist())
                    varied_values.extend(
                        attrs[:, attributes.index(attr)].cpu().numpy().tolist()
                    )
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
        print('x_vals '+str(x_vals))
        print('mean_true '+str(mean_true))
        print('mean_pred '+str(mean_pred))
        ax = axs[i]
        ax.plot(x_vals, mean_true, 'o-', color='black', label='Ground Truth')
        ax.plot(x_vals, mean_pred, 's--', color='tab:blue', label='Predicted')
        ax.fill_between(x_vals, mean_pred - std_pred, mean_pred + std_pred, color='tab:blue', alpha=0.2)
        ax.set_title(attr,fontsize=12)
        ax.set_xlabel("Attribute Value (normalized 0–1)",fontsize=12)
        ax.set_ylabel("Target Value (normalized 0→1)",fontsize=12)
        ax.grid(True)
        if i == 0:
            ax.legend()

    plt.suptitle("Model Target Sensitivity", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


import torch
import matplotlib.pyplot as plt
def visualize_roundness_cs_interaction(model, args, img_size):
    model.eval()
    roundness_values = [1, 2, 3, 4, 5]
    seeds =  list(range(0, 100))#[random.randint(1, 1000) for _ in range(100)]
    # store results separated by CS
    results = {0: {"true": [], "pred": []}, 1: {"true": [], "pred": []}}
    for s in seeds:
        dl = create_attribute_dataloader(
            attribute_name="roundness",
            base_seed=s,
            img_size=img_size,
            batch_size=1,
            fix_all_but_r_and_is=True
        )
        true_targets_cs = {0: [], 1: []}
        pred_targets_cs = {0: [], 1: []}
        for img, mask, attrs, target, _ in dl:
            cs_val = int(attrs[0, -1].item())
            img = img.to("cuda", dtype=torch.float)
            with torch.no_grad():
                if args.base_model == "ViT":
                    pred_outs, _ = model(img)
                    if args.dataset == "FunnyNodules":
                        preds = pred_outs[:, 6:]
                elif args.base_model in ["ResNetMT","DenseNetMT"]:
                    img = img.repeat(1, 3, 1, 1)
                    outputs = model(img)
                    preds = outputs[:, 0]
                else:
                    if args.dataset == "FunnyNodules":
                        preds, _, _, _ = model(img)
                pred = preds.squeeze().cpu().item()
                pred_targets_cs[cs_val].append(pred)
                true_targets_cs[cs_val].append(target.item())
        for cs in [0, 1]:
            if len(true_targets_cs[cs]) > 0:
                results[cs]["true"].append(true_targets_cs[cs])
                results[cs]["pred"].append(pred_targets_cs[cs])
    plt.figure(figsize=(7,5))
    for cs in [0, 1]:
        all_true = np.array(results[cs]["true"])
        all_pred = np.array(results[cs]["pred"])
        mean_true = all_true.mean(axis=0)
        mean_pred = all_pred.mean(axis=0)
        std_pred = all_pred.std(axis=0)
        color_gt = 'black' if cs==0 else 'gray'
        color_pred = 'tab:blue' if cs==0 else 'tab:orange'
        plt.plot(roundness_values, mean_true, 'o-', color=color_gt, label=f"GT is={cs}")
        plt.plot(roundness_values, mean_pred, 's--', color=color_pred, label=f"Pred is={cs}")
        plt.fill_between(roundness_values, mean_pred - std_pred, mean_pred + std_pred,
                         color=color_pred, alpha=0.2)
        print('visualize_roundness_is_interaction')
        print('is '+str(cs))
        print('roundness_values '+str(roundness_values))
        print('mean_true '+str(mean_true))
        print('mean_pred '+str(mean_pred))

    plt.xlabel("Roundness")
    plt.ylabel("Target Value (normalized 0→1)")
    plt.title("Effect of Roundness on Target conditioned on IS with other attributes=3")
    plt.grid(True)
    plt.legend()
    plt.show()

