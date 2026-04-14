import os

import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def test_time_intervention(testmodel, val_data_loader, test_data_loader, epoch,
                           prototypefoldername, args):
    print("test-time intervention")
    prime_proto_latent_vectors = get_prime_protos_test_time_intervention(testmodel, val_data_loader, epoch,
                                                                         prototypefoldername, args)
    print("prime_promo_latent_vectors")
    print(len(prime_proto_latent_vectors))
    print(prime_proto_latent_vectors[0].shape)
    attr_order = get_attri_order_test_time_intervention(prime_proto_latent_vectors, testmodel, val_data_loader, epoch,
                                                        prototypefoldername, args)
    print("attribute order for test time intervention: "+str(attr_order))
    interference_test_time_intervention(prime_proto_latent_vectors, attr_order,
                                        testmodel,test_data_loader, epoch, prototypefoldername, args)



def interference_test_time_intervention(prime_proto_latent_vectors, attr_order, testmodel, data_loader, epoch,
                                        prototypefoldername, args):
    """
    calculate target accuracy when intervening with correct proto latent vectors of attributes in attr_order
    output target and PE_target
    output when intervening on 1, 1+2, 1+2+3, ... attributes in attr_order
    """
    print("interference with test time intervention")
    testmodel.eval()
    if args.dataset == "LIDC":
        num_attributes = 8
    elif args.dataset == "derm7pt":
        num_attributes = 7

    correct_mal = 0
    correct_PE_mal = 0
    correct_mal_with_corrected_lv_if_wrong = torch.zeros(num_attributes)
    correct_PE_mal_with_corrected_lv_if_wrong = torch.zeros(num_attributes)

    with torch.no_grad():
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

            if args.base_model == "ViT":
                dists_to_protos = testmodel.getDistance(x)
                pred_outs, pred_mask = testmodel(x)
                if args.dataset == "LIDC":
                    pred_attr = pred_outs[:, :8]
                    pred_mal = pred_outs[:, 8:]
                elif args.dataset == "derm7pt":
                    pred_attr = []
                    pred_attr.append(pred_outs[:, 0:3])
                    pred_attr.append(pred_outs[:, 3:5])
                    pred_attr.append(pred_outs[:, 5:8])
                    pred_attr.append(pred_outs[:, 8:11])
                    pred_attr.append(pred_outs[:, 11:14])
                    pred_attr.append(pred_outs[:, 14:17])
                    pred_attr.append(pred_outs[:, 17:19])
                    pred_mal = pred_outs[:, 19:]
            else:
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
            if args.dataset in ["LIDC", "derm7pt"]:
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
                x_ex = torch.zeros((x.shape[0], len(min_proto_idx_allcaps), testmodel.protodigis0.shape[-2],
                                    testmodel.protodigis0.shape[-1]))
            else:
                x_ex = torch.zeros((x.shape[0], len(min_proto_idx_allcaps), testmodel.protodigis0.shape[-1]))
            if args.dataset == "CUB":
                attr_classes = [5, 5, 4, 5, 4]
                attrLabelsSamples_protos = [[[0] * length for length in attr_classes]] * len(y_attributes[0])
                attrLabelsSamples_protos_max = [[[0] * length for length in attr_classes]] * len(y_attributes[0])
            else:
                attrLabelsSamples_protos = torch.zeros_like(y_attributes)
                attrLabelsSamples_protos_max = torch.zeros_like(y_attributes)
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
                            proto_scores_str = image_name.split("[")[1].split("]")[0].split(", ")
                            proto_scores = torch.tensor(np.array(proto_scores_str).astype(np.float64))
                            attrLabelsSamples_protos[sai, capsule_idx] = proto_scores[capsule_idx]
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

            if args.dataset in ["LIDC", "derm7pt"]:
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

                elif args.dataset == "derm7pt":
                    mal_confusion_matrix = confusion_matrix(y_mal.cpu().detach().numpy(),
                                                            torch.argmax(pred_mal, dim=-1).cpu().detach().numpy(),
                                                            labels=[0, 1, 2, 3, 4])
                else:
                    mal_confusion_matrix = confusion_matrix(np.argmax(y_mal.cpu().detach().numpy(), axis=1) + 1,
                                                            np.argmax(pred_mal.cpu().detach().numpy(), axis=1) + 1,
                                                            labels=[1, 2, 3, 4, 5])

                if args.dataset == "derm7pt":
                    mal_correct_within_one = sum(np.diagonal(mal_confusion_matrix, offset=0))
                    print("confusion matrix target")
                    print(mal_confusion_matrix)
                elif args.dataset == "LIDC":
                    mal_correct_within_one = sum(np.diagonal(mal_confusion_matrix, offset=0)) + \
                                             sum(np.diagonal(mal_confusion_matrix, offset=1)) + \
                                             sum(np.diagonal(mal_confusion_matrix, offset=-1))
                correct_mal += mal_correct_within_one

            if args.dataset in ["LIDC", "derm7pt"]:
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

                elif args.dataset == "derm7pt":
                    mal_confusion_matrix = confusion_matrix(y_mal.cpu().detach().numpy(),
                                                            torch.argmax(pred_p, dim=-1).cpu().detach().numpy(),
                                                            labels=[0, 1, 2, 3, 4])
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
                correct_PE_mal += mal_correct_within_one

                if args.dataset == "derm7pt":
                    x_ex_intervened_PE = x_ex
                    x_ex_intervened = testmodel.attribute_lv(x)
                    for attri_i in attr_order:
                        for sai in range(y_attributes.shape[0]):
                            lvs_intervention = testmodel.attribute_lv(x[sai])[0]
                            if not y_attributes[sai, attri_i] == torch.argmax(pred_attr[attri_i][sai], dim=-1):
                                lvs_intervention[attri_i] = prime_proto_latent_vectors[attri_i][
                                    y_attributes[sai, attri_i]]
                            x_ex_intervened[sai, attri_i] = lvs_intervention[attri_i]
                            x_ex_intervened_PE[sai, attri_i] = lvs_intervention[attri_i]
                        pred_intervened = testmodel.forwardprotodigis(
                            x_ex=x_ex_intervened.to("cuda", dtype=torch.float))
                        pred_intervened_PE = testmodel.forwardprotodigis(
                            x_ex=x_ex_intervened_PE.to("cuda", dtype=torch.float))
                        if args.base_model == "ViT":
                            mal_confusion_matrix = confusion_matrix(y_mal.cpu().detach().numpy(),
                                                                    torch.argmax(pred_intervened,
                                                                                 dim=-1).cpu().detach().numpy(),
                                                                    labels=[0, 1, 2, 3, 4])
                            mal_confusion_matrix_PE = confusion_matrix(y_mal.cpu().detach().numpy(),
                                                                       torch.argmax(pred_intervened_PE,
                                                                                    dim=-1).cpu().detach().numpy(),
                                                                       labels=[0, 1, 2, 3, 4])
                        mal_correct_within_one = sum(np.diagonal(mal_confusion_matrix, offset=0))
                        mal_correct_within_one_PE = sum(np.diagonal(mal_confusion_matrix_PE, offset=0))
                        correct_mal_with_corrected_lv_if_wrong[attri_i] += mal_correct_within_one
                        correct_PE_mal_with_corrected_lv_if_wrong[attri_i] += mal_correct_within_one_PE

                elif args.dataset == "LIDC":
                    y_attributes[:, [0, 3, 4, 5, 6, 7]] *= 4
                    y_attributes[:, 1] *= 3
                    y_attributes[:, 2] *= 5
                    y_attributes += 1
                    pred_attr[:, [0, 3, 4, 5, 6, 7]] *= 4
                    pred_attr[:, 1] *= 3
                    pred_attr[:, 2] *= 5
                    pred_attr += 1
                    x_ex_intervened_PE = x_ex
                    x_ex_intervened = testmodel.attribute_lv(x)
                    for attri_i in attr_order:
                        for sai in range(y_attributes.shape[0]):
                            lvs_intervention = testmodel.attribute_lv(x[sai])[0]
                            if (np.rint(y_attributes[sai, attri_i].cpu().detach().numpy()) -
                                np.rint(pred_attr[sai, attri_i].cpu().detach().numpy())) > 1:
                                lvs_intervention[attri_i] = prime_proto_latent_vectors[attri_i][
                                    int(np.rint(y_attributes[sai, attri_i].cpu().detach().numpy())) - 1]
                            x_ex_intervened[sai, attri_i] = lvs_intervention[attri_i]
                            x_ex_intervened_PE[sai, attri_i] = lvs_intervention[attri_i]
                        pred_intervened = testmodel.forwardprotodigis(
                            x_ex=x_ex_intervened.to("cuda", dtype=torch.float))
                        pred_intervened_PE = testmodel.forwardprotodigis(
                            x_ex=x_ex_intervened_PE.to("cuda", dtype=torch.float))
                        if args.base_model == "ViT":
                            pred_intervened *= 4
                            pred_intervened = torch.round(pred_intervened)
                            mal_confusion_matrix = confusion_matrix(
                                np.argmax(y_mal.cpu().detach().numpy(), axis=1) + 1,
                                pred_intervened.cpu().detach().numpy() + 1,
                                labels=[1, 2, 3, 4, 5])
                            pred_intervened_PE *= 4
                            pred_intervened_PE = torch.round(pred_intervened_PE)
                            mal_confusion_matrix_PE = confusion_matrix(
                                np.argmax(y_mal.cpu().detach().numpy(), axis=1) + 1,
                                pred_intervened_PE.cpu().detach().numpy() + 1,
                                labels=[1, 2, 3, 4, 5])
                        mal_correct_within_one = sum(np.diagonal(mal_confusion_matrix, offset=0)) + \
                                                 sum(np.diagonal(mal_confusion_matrix, offset=1)) + \
                                                 sum(np.diagonal(mal_confusion_matrix, offset=-1))
                        mal_correct_within_one_PE = sum(np.diagonal(mal_confusion_matrix_PE, offset=0)) + \
                                                    sum(np.diagonal(mal_confusion_matrix_PE, offset=1)) + \
                                                    sum(np.diagonal(mal_confusion_matrix_PE, offset=-1))
                        correct_mal_with_corrected_lv_if_wrong[attri_i] += mal_correct_within_one
                        correct_PE_mal_with_corrected_lv_if_wrong[attri_i] += mal_correct_within_one_PE
    print("Results test time interference:")
    print("origtarget         origtargetPE       ttitarget                                                                ttitargetPE)")
    print(correct_mal / len(data_loader.dataset), \
          correct_PE_mal / len(data_loader.dataset), \
          correct_mal_with_corrected_lv_if_wrong[attr_order] / len(data_loader.dataset), \
          correct_PE_mal_with_corrected_lv_if_wrong[attr_order] / len(data_loader.dataset))


def get_prime_protos_test_time_intervention(testmodel, data_loader, epoch,
                                            prototypefoldername, args):
    """
    return list of attribute order and prime-proto latent vectors
    """
    print("get latent vectors from prime prototypes")
    testmodel.eval()
    if args.dataset in ["LIDC", "derm7pt"]:
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

    with torch.no_grad():
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

            if args.base_model == "ViT":
                dists_to_protos = testmodel.getDistance(x)
            else:
                _, _, _, dists_to_protos = testmodel(x)

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
            if args.dataset in ["LIDC", "derm7pt"]:
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
            for sai in range(x.shape[0]):
                for capsule_idx in range(len(min_proto_idx_allcaps)):
                    usedProtos_allcpsi[capsule_idx][
                        min_proto_idx_allcaps[capsule_idx][sai][0], min_proto_idx_allcaps[capsule_idx][sai][1]] += 1

    torch.set_printoptions(profile="full")
    print("Prototypes: how many times were prototypes used:")
    print(len(usedProtos_allcpsi))
    print(usedProtos_allcpsi[0].shape)
    print(usedProtos_allcpsi)

    print("get prime_promo_laten_vectors")
    print("von hier aus nur noch VIT unterstützt")
    prime_promo_latent_vectors = []
    for i in range(len(usedProtos_allcpsi)):
        prime_promo_latent_vectors.append(
            torch.zeros((testmodel.protodigis_list[i].shape[0],
                         testmodel.protodigis0.shape[-2],
                         testmodel.protodigis0.shape[-1])))
        print(prime_promo_latent_vectors[i].shape)

    for attris in range(len(usedProtos_allcpsi)):
        for attri_value in range(len(usedProtos_allcpsi[attris])):
            print(usedProtos_allcpsi[attris][attri_value])
            prime_promo_idx = torch.argmax(usedProtos_allcpsi[attris][attri_value])
            print(prime_promo_idx)
            prime_promo_latent_vector = testmodel.protodigis_list[attris][attri_value, prime_promo_idx]
            print(prime_promo_latent_vector.shape)
            prime_promo_latent_vectors[attris][attri_value] = prime_promo_latent_vector
    return prime_promo_latent_vectors


def get_attri_order_test_time_intervention(prime_proto_latent_vectors, testmodel, data_loader, epoch,
                                           prototypefoldername, args):
    """
    return list of attribute order and prime-proto latent vectors
    """
    print("get order of most-inpactful attributes")
    testmodel.eval()
    if args.dataset == "LIDC":
        num_attributes = 8
    elif args.dataset == "derm7pt":
        num_attributes = 7

    correct_mal = 0
    correct_PE_mal = 0
    correct_mal_with_corrected_lv_if_wrong = torch.zeros(num_attributes)
    correct_PE_mal_with_corrected_lv_if_wrong = torch.zeros(num_attributes)

    with torch.no_grad():
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

            if args.base_model == "ViT":
                dists_to_protos = testmodel.getDistance(x)
                pred_outs, pred_mask = testmodel(x)
                if args.dataset == "LIDC":
                    pred_attr = pred_outs[:, :8]
                    pred_mal = pred_outs[:, 8:]
                elif args.dataset == "derm7pt":
                    pred_attr = []
                    pred_attr.append(pred_outs[:, 0:3])
                    pred_attr.append(pred_outs[:, 3:5])
                    pred_attr.append(pred_outs[:, 5:8])
                    pred_attr.append(pred_outs[:, 8:11])
                    pred_attr.append(pred_outs[:, 11:14])
                    pred_attr.append(pred_outs[:, 14:17])
                    pred_attr.append(pred_outs[:, 17:19])
                    pred_mal = pred_outs[:, 19:]
            else:
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
            if args.dataset in ["LIDC", "derm7pt"]:
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
                x_ex = torch.zeros((x.shape[0], len(min_proto_idx_allcaps), testmodel.protodigis0.shape[-2],
                                    testmodel.protodigis0.shape[-1]))
            else:
                x_ex = torch.zeros((x.shape[0], len(min_proto_idx_allcaps), testmodel.protodigis0.shape[-1]))
            if args.dataset == "CUB":
                attr_classes = [5, 5, 4, 5, 4]
                attrLabelsSamples_protos = [[[0] * length for length in attr_classes]] * len(y_attributes[0])
                attrLabelsSamples_protos_max = [[[0] * length for length in attr_classes]] * len(y_attributes[0])
            else:
                attrLabelsSamples_protos = torch.zeros_like(y_attributes)
                attrLabelsSamples_protos_max = torch.zeros_like(y_attributes)
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
                            proto_scores_str = image_name.split("[")[1].split("]")[0].split(", ")
                            proto_scores = torch.tensor(np.array(proto_scores_str).astype(np.float64))
                            attrLabelsSamples_protos[sai, capsule_idx] = proto_scores[capsule_idx]
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

            if args.dataset in ["LIDC", "derm7pt"]:
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

                elif args.dataset == "derm7pt":
                    mal_confusion_matrix = confusion_matrix(y_mal.cpu().detach().numpy(),
                                                            torch.argmax(pred_mal, dim=-1).cpu().detach().numpy(),
                                                            labels=[0, 1, 2, 3, 4])
                else:
                    mal_confusion_matrix = confusion_matrix(np.argmax(y_mal.cpu().detach().numpy(), axis=1) + 1,
                                                            np.argmax(pred_mal.cpu().detach().numpy(), axis=1) + 1,
                                                            labels=[1, 2, 3, 4, 5])

                if args.dataset == "derm7pt":
                    mal_correct_within_one = sum(np.diagonal(mal_confusion_matrix, offset=0))
                    print("confusion matrix target")
                    print(mal_confusion_matrix)
                elif args.dataset == "LIDC":
                    mal_correct_within_one = sum(np.diagonal(mal_confusion_matrix, offset=0)) + \
                                             sum(np.diagonal(mal_confusion_matrix, offset=1)) + \
                                             sum(np.diagonal(mal_confusion_matrix, offset=-1))
                correct_mal += mal_correct_within_one

            if args.dataset in ["LIDC", "derm7pt"]:
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

                elif args.dataset == "derm7pt":
                    mal_confusion_matrix = confusion_matrix(y_mal.cpu().detach().numpy(),
                                                            torch.argmax(pred_p, dim=-1).cpu().detach().numpy(),
                                                            labels=[0, 1, 2, 3, 4])
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
                correct_PE_mal += mal_correct_within_one



                if args.dataset == "derm7pt":
                    for attri_i in range(7):
                        x_ex_intervened_PE = x_ex
                        x_ex_intervened = testmodel.attribute_lv(x)
                        for sai in range(y_attributes.shape[0]):
                            lvs_intervention = testmodel.attribute_lv(x[sai])[0]
                            if not y_attributes[sai, attri_i] == torch.argmax(pred_attr[attri_i][sai], dim=-1):
                                lvs_intervention[attri_i] = prime_proto_latent_vectors[attri_i][y_attributes[sai, attri_i]]
                            x_ex_intervened[sai, attri_i] = lvs_intervention[attri_i]
                            x_ex_intervened_PE[sai, attri_i] = lvs_intervention[attri_i]
                        pred_intervened = testmodel.forwardprotodigis(x_ex=x_ex_intervened.to("cuda", dtype=torch.float))
                        pred_intervened_PE = testmodel.forwardprotodigis(x_ex=x_ex_intervened_PE.to("cuda", dtype=torch.float))
                        if args.base_model == "ViT":
                            mal_confusion_matrix = confusion_matrix(y_mal.cpu().detach().numpy(),
                                                                    torch.argmax(pred_intervened,
                                                                                 dim=-1).cpu().detach().numpy(),
                                                                    labels=[0, 1, 2, 3, 4])
                            mal_confusion_matrix_PE = confusion_matrix(y_mal.cpu().detach().numpy(),
                                                                    torch.argmax(pred_intervened_PE,
                                                                                 dim=-1).cpu().detach().numpy(),
                                                                    labels=[0, 1, 2, 3, 4])
                        mal_correct_within_one = sum(np.diagonal(mal_confusion_matrix, offset=0))
                        mal_correct_within_one_PE = sum(np.diagonal(mal_confusion_matrix_PE, offset=0))
                        correct_mal_with_corrected_lv_if_wrong[attri_i] += mal_correct_within_one
                        correct_PE_mal_with_corrected_lv_if_wrong[attri_i] += mal_correct_within_one_PE
                        #     todo test

                elif args.dataset == "LIDC":
                    y_attributes[:, [0, 3, 4, 5, 6, 7]] *= 4
                    y_attributes[:, 1] *= 3
                    y_attributes[:, 2] *= 5
                    y_attributes += 1
                    pred_attr[:, [0, 3, 4, 5, 6, 7]] *= 4
                    pred_attr[:, 1] *= 3
                    pred_attr[:, 2] *= 5
                    pred_attr += 1
                    for attri_i in range(y_attributes.shape[1]):
                        x_ex_intervened_PE = x_ex
                        x_ex_intervened = testmodel.attribute_lv(x)
                        for sai in range(y_attributes.shape[0]):
                            lvs_intervention = testmodel.attribute_lv(x[sai])[0]
                            if (np.rint(y_attributes[sai, attri_i].cpu().detach().numpy()) -
                                np.rint(pred_attr[sai, attri_i].cpu().detach().numpy())) > 1:
                                lvs_intervention[attri_i] = prime_proto_latent_vectors[attri_i][int(np.rint(y_attributes[sai, attri_i].cpu().detach().numpy()))-1]
                            x_ex_intervened[sai, attri_i] = lvs_intervention[attri_i]
                            x_ex_intervened_PE[sai, attri_i] = lvs_intervention[attri_i]
                        pred_intervened = testmodel.forwardprotodigis(x_ex=x_ex_intervened.to("cuda", dtype=torch.float))
                        pred_intervened_PE = testmodel.forwardprotodigis(x_ex=x_ex_intervened_PE.to("cuda", dtype=torch.float))
                        if args.base_model == "ViT":
                            pred_intervened *= 4
                            pred_intervened = torch.round(pred_intervened)
                            mal_confusion_matrix = confusion_matrix(
                                np.argmax(y_mal.cpu().detach().numpy(), axis=1) + 1,
                                pred_intervened.cpu().detach().numpy() + 1,
                                labels=[1, 2, 3, 4, 5])
                            pred_intervened_PE *= 4
                            pred_intervened_PE = torch.round(pred_intervened_PE)
                            mal_confusion_matrix_PE = confusion_matrix(
                                np.argmax(y_mal.cpu().detach().numpy(), axis=1) + 1,
                                pred_intervened_PE.cpu().detach().numpy() + 1,
                                labels=[1, 2, 3, 4, 5])
                        mal_correct_within_one = sum(np.diagonal(mal_confusion_matrix, offset=0)) + \
                                                 sum(np.diagonal(mal_confusion_matrix, offset=1)) + \
                                                 sum(np.diagonal(mal_confusion_matrix, offset=-1))
                        mal_correct_within_one_PE = sum(np.diagonal(mal_confusion_matrix_PE, offset=0)) + \
                                                 sum(np.diagonal(mal_confusion_matrix_PE, offset=1)) + \
                                                 sum(np.diagonal(mal_confusion_matrix_PE, offset=-1))
                        correct_mal_with_corrected_lv_if_wrong[attri_i] += mal_correct_within_one
                        correct_PE_mal_with_corrected_lv_if_wrong[attri_i] += mal_correct_within_one_PE



    print(correct_mal / len(data_loader.dataset), \
          correct_PE_mal / len(data_loader.dataset), \
          correct_mal_with_corrected_lv_if_wrong / len(data_loader.dataset), \
          correct_PE_mal_with_corrected_lv_if_wrong / len(data_loader.dataset))
    attr_order = torch.argsort(correct_mal_with_corrected_lv_if_wrong / len(data_loader.dataset), descending=True)
    print(attr_order)
    return attr_order.tolist()


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append((index % dim).item())
        index = index // dim
    return tuple(reversed(out))
