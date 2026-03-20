"""
Proto-Caps model

Author: Luisa Gallée, Github: `https://github.com/XRad-Ulm/Proto-Caps`
"""
import numpy as np
import torch
from torch import nn
from capsulelayers import DenseCapsule, PrimaryCapsule

class ProtoCapsNet(nn.Module):
    def __init__(self, input_size, numcaps, routings, out_dim_caps, activation_fn, threeD, numProtos, args):
        super(ProtoCapsNet, self).__init__()
        self.input_size = input_size
        self.numcaps = numcaps#+4
        self.routings = routings
        self.out_dim_caps = out_dim_caps
        if args.dataset == "LIDC":
            if args.ordinal_target:
                self.numclasses = 4
            else:
                self.numclasses = 5
        elif args.dataset == "derm7pt":
            self.numclasses = 5
        elif args.dataset == "Chexbert":
            self.numclasses = 1
        if args.dataset == "FunnyNodules":
            self.numclasses = 1
        self.threeD = threeD
        self.numProtos = numProtos
        self.args = args

        # Layer 1: Just a conventional Conv2D layer
        if self.threeD:
            self.conv1 = nn.Conv3d(input_size[0], 256, kernel_size=9, stride=1, padding=0)
            self.primarycaps = PrimaryCapsule(256, 256, 8, kernel_size=3, threeD=self.threeD, stride=2, padding=0)
        else:
            self.conv1 = nn.Conv2d(input_size[0], 256, kernel_size=9, stride=1, padding=0)
            self.primarycaps = PrimaryCapsule(256, 256, 8, kernel_size=9, threeD=self.threeD, stride=2, padding=0)


        if self.threeD:
            self.digitcaps = DenseCapsule(in_num_caps=11616, in_dim_caps=8,
                                          out_num_caps=self.numcaps, out_dim_caps=out_dim_caps, routings=routings,
                                          activation_fn=activation_fn)
        else:
            if args.dataset == "LIDC":
                print("use this digitcaps")
                self.digitcaps = DenseCapsule(in_num_caps=32 * 8 * 8, in_dim_caps=8,
                                              out_num_caps=self.numcaps, out_dim_caps=out_dim_caps, routings=routings,
                                              activation_fn=activation_fn)
            elif args.dataset == "Chexbert":
                self.digitcaps = DenseCapsule(in_num_caps=18432, in_dim_caps=8,
                                              out_num_caps=self.numcaps, out_dim_caps=out_dim_caps, routings=routings,
                                              activation_fn=activation_fn)
            elif args.dataset == "derm7pt":
                self.digitcaps = DenseCapsule(in_num_caps=2048, in_dim_caps=8,
                                              out_num_caps=self.numcaps, out_dim_caps=out_dim_caps, routings=routings,
                                              activation_fn=activation_fn)
            elif args.dataset == "FunnyNodules":
                self.digitcaps = DenseCapsule(in_num_caps=346112, in_dim_caps=8,
                                              out_num_caps=self.numcaps, out_dim_caps=out_dim_caps, routings=routings,
                                              activation_fn=activation_fn)

        # Decoder network.
        if self.threeD:
            self.decoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.numcaps * out_dim_caps, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, input_size[0] * input_size[1] * input_size[2] * input_size[3]),
                nn.Sigmoid()
            )
        else:
            self.decoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.numcaps * out_dim_caps, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, input_size[0] * input_size[1] * input_size[2]),
                nn.Sigmoid()
            )

        # Prediction layers

        if args.dataset == "LIDC":
            if args.ordinal_target:
                self.predOutLayers = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(in_features=self.numcaps * out_dim_caps, out_features=self.numclasses),
                    nn.Sigmoid()
                )
            else:
                self.predOutLayers = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(in_features=self.numcaps * out_dim_caps, out_features=self.numclasses),
                    nn.Softmax(dim=-1)
                )
        elif args.dataset == "Chexbert":
            self.predOutLayers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=self.numcaps * out_dim_caps, out_features=self.numclasses),
                nn.Sigmoid()
            )
        elif args.dataset == "derm7pt":
            self.predOutLayers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=self.numcaps * out_dim_caps, out_features=self.numclasses),
                nn.Softmax(dim=-1)
            )
        elif args.dataset == "FunnyNodules":
            self.predOutLayers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=self.numcaps * out_dim_caps, out_features=self.numclasses),
                nn.Sigmoid()
            )

        self.relu = nn.ReLU()

        # Attribute layers
        if not args.dataset == "derm7pt":
            self.attrOutLayer0 = nn.Sequential(
                nn.Linear(in_features=out_dim_caps, out_features=1),
                nn.Sigmoid()
            )
            self.attrOutLayer1 = nn.Sequential(
                nn.Linear(in_features=out_dim_caps, out_features=1),
                nn.Sigmoid()
            )
            self.attrOutLayer2 = nn.Sequential(
                nn.Linear(in_features=out_dim_caps, out_features=1),
                nn.Sigmoid()
            )
            self.attrOutLayer3 = nn.Sequential(
                nn.Linear(in_features=out_dim_caps, out_features=1),
                nn.Sigmoid()
            )
            self.attrOutLayer4 = nn.Sequential(
                nn.Linear(in_features=out_dim_caps, out_features=1),
                nn.Sigmoid()
            )
            self.attrOutLayer5 = nn.Sequential(
                nn.Linear(in_features=out_dim_caps, out_features=1),
                nn.Sigmoid()
            )
            if not args.dataset == "FunnyNodules":
                self.attrOutLayer6 = nn.Sequential(
                    nn.Linear(in_features=out_dim_caps, out_features=1),
                    nn.Sigmoid()
                )
                self.attrOutLayer7 = nn.Sequential(
                    nn.Linear(in_features=out_dim_caps, out_features=1),
                    nn.Sigmoid()
                )
            if args.dataset == "Chexbert":
                self.attrOutLayer8 = nn.Sequential(
                    nn.Linear(in_features=out_dim_caps, out_features=1),
                    nn.Sigmoid()
                )
                self.attrOutLayer9 = nn.Sequential(
                    nn.Linear(in_features=out_dim_caps, out_features=1),
                    nn.Sigmoid()
                )
                self.attrOutLayer10 = nn.Sequential(
                    nn.Linear(in_features=out_dim_caps, out_features=1),
                    nn.Sigmoid()
                )
                self.attrOutLayer11 = nn.Sequential(
                    nn.Linear(in_features=out_dim_caps, out_features=1),
                    nn.Sigmoid()
                )
                self.attrOutLayer12 = nn.Sequential(
                    nn.Linear(in_features=out_dim_caps, out_features=1),
                    nn.Sigmoid()
                )
        else:
            self.attrOutLayer0 = nn.Sequential(
                nn.Linear(in_features=out_dim_caps, out_features=3),
                nn.Softmax(-1)
            )
            self.attrOutLayer1 = nn.Sequential(
                nn.Linear(in_features=out_dim_caps, out_features=2),
                nn.Softmax(-1)
            )
            self.attrOutLayer2 = nn.Sequential(
                nn.Linear(in_features=out_dim_caps, out_features=3),
                nn.Softmax(-1)
            )
            self.attrOutLayer3 = nn.Sequential(
                nn.Linear(in_features=out_dim_caps, out_features=3),
                nn.Softmax(-1)
            )
            self.attrOutLayer4 = nn.Sequential(
                nn.Linear(in_features=out_dim_caps, out_features=3),
                nn.Softmax(-1)
            )
            self.attrOutLayer5 = nn.Sequential(
                nn.Linear(in_features=out_dim_caps, out_features=3),
                nn.Softmax(-1)
            )
            self.attrOutLayer6 = nn.Sequential(
                nn.Linear(in_features=out_dim_caps, out_features=2),
                nn.Softmax(-1)
            )

        # Prototype vectors
        if args.dataset == "LIDC":
            self.protodigis0 = nn.Parameter(torch.rand((5, self.numProtos, out_dim_caps)), requires_grad=True)
            self.protodigis1 = nn.Parameter(torch.rand((4, self.numProtos, out_dim_caps)), requires_grad=True)
            self.protodigis2 = nn.Parameter(torch.rand((6, self.numProtos, out_dim_caps)), requires_grad=True)
            self.protodigis3 = nn.Parameter(torch.rand((5, self.numProtos, out_dim_caps)), requires_grad=True)
            self.protodigis4 = nn.Parameter(torch.rand((5, self.numProtos, out_dim_caps)), requires_grad=True)
            self.protodigis5 = nn.Parameter(torch.rand((5, self.numProtos, out_dim_caps)), requires_grad=True)
            self.protodigis6 = nn.Parameter(torch.rand((5, self.numProtos, out_dim_caps)), requires_grad=True)
            self.protodigis7 = nn.Parameter(torch.rand((5, self.numProtos, out_dim_caps)), requires_grad=True)
            self.protodigis_list = [self.protodigis0, self.protodigis1, self.protodigis2, self.protodigis3,
                                    self.protodigis4, self.protodigis5, self.protodigis6, self.protodigis7]
            if self.numcaps != numcaps:
                print("more capsules than attributes")
                self.protodigis8 = nn.Parameter(torch.rand((1, self.numProtos, out_dim_caps)), requires_grad=True)
                self.protodigis9 = nn.Parameter(torch.rand((1, self.numProtos, out_dim_caps)), requires_grad=True)
                self.protodigis10 = nn.Parameter(torch.rand((1, self.numProtos, out_dim_caps)), requires_grad=True)
                self.protodigis11 = nn.Parameter(torch.rand((1, self.numProtos, out_dim_caps)), requires_grad=True)
                self.protodigis12 = nn.Parameter(torch.rand((1, self.numProtos, out_dim_caps)), requires_grad=True)
                self.protodigis13 = nn.Parameter(torch.rand((1, self.numProtos, out_dim_caps)), requires_grad=True)
                self.protodigis14 = nn.Parameter(torch.rand((1, self.numProtos, out_dim_caps)), requires_grad=True)
                self.protodigis15 = nn.Parameter(torch.rand((1, self.numProtos, out_dim_caps)), requires_grad=True)
                self.protodigis_list.append(self.protodigis8)
                self.protodigis_list.append(self.protodigis9)
                self.protodigis_list.append(self.protodigis10)
                self.protodigis_list.append(self.protodigis11)
                print(len(self.protodigis_list))
        elif args.dataset == "Chexbert":
            self.protodigis0 = nn.Parameter(torch.rand((2, self.numProtos, out_dim_caps)), requires_grad=True)
            self.protodigis1 = nn.Parameter(torch.rand((2, self.numProtos, out_dim_caps)), requires_grad=True)
            self.protodigis2 = nn.Parameter(torch.rand((2, self.numProtos, out_dim_caps)), requires_grad=True)
            self.protodigis3 = nn.Parameter(torch.rand((2, self.numProtos, out_dim_caps)), requires_grad=True)
            self.protodigis4 = nn.Parameter(torch.rand((2, self.numProtos, out_dim_caps)), requires_grad=True)
            self.protodigis5 = nn.Parameter(torch.rand((2, self.numProtos, out_dim_caps)), requires_grad=True)
            self.protodigis6 = nn.Parameter(torch.rand((2, self.numProtos, out_dim_caps)), requires_grad=True)
            self.protodigis7 = nn.Parameter(torch.rand((2, self.numProtos, out_dim_caps)), requires_grad=True)
            self.protodigis8 = nn.Parameter(torch.rand((2, self.numProtos, out_dim_caps)), requires_grad=True)
            self.protodigis9 = nn.Parameter(torch.rand((2, self.numProtos, out_dim_caps)), requires_grad=True)
            self.protodigis10 = nn.Parameter(torch.rand((2, self.numProtos, out_dim_caps)), requires_grad=True)
            self.protodigis11 = nn.Parameter(torch.rand((2, self.numProtos, out_dim_caps)), requires_grad=True)
            self.protodigis12 = nn.Parameter(torch.rand((2, self.numProtos, out_dim_caps)), requires_grad=True)
            self.protodigis_list = [self.protodigis0, self.protodigis1, self.protodigis2, self.protodigis3,
                                    self.protodigis4, self.protodigis5, self.protodigis6, self.protodigis7,
                                    self.protodigis8, self.protodigis9, self.protodigis10, self.protodigis11,
                                    self.protodigis12]
        elif args.dataset == "derm7pt":
            self.protodigis0 = nn.Parameter(torch.rand((3, self.numProtos, out_dim_caps)), requires_grad=True)
            self.protodigis1 = nn.Parameter(torch.rand((2, self.numProtos, out_dim_caps)), requires_grad=True)
            self.protodigis2 = nn.Parameter(torch.rand((3, self.numProtos, out_dim_caps)), requires_grad=True)
            self.protodigis3 = nn.Parameter(torch.rand((3, self.numProtos, out_dim_caps)), requires_grad=True)
            self.protodigis4 = nn.Parameter(torch.rand((3, self.numProtos, out_dim_caps)), requires_grad=True)
            self.protodigis5 = nn.Parameter(torch.rand((3, self.numProtos, out_dim_caps)), requires_grad=True)
            self.protodigis6 = nn.Parameter(torch.rand((2, self.numProtos, out_dim_caps)), requires_grad=True)
            self.protodigis_list = [self.protodigis0, self.protodigis1, self.protodigis2, self.protodigis3,
                                    self.protodigis4, self.protodigis5, self.protodigis6]
        elif args.dataset == "FunnyNodules":
            self.protodigis0 = nn.Parameter(torch.rand((5, self.numProtos, out_dim_caps)), requires_grad=True)
            self.protodigis1 = nn.Parameter(torch.rand((5, self.numProtos, out_dim_caps)), requires_grad=True)
            self.protodigis2 = nn.Parameter(torch.rand((5, self.numProtos, out_dim_caps)), requires_grad=True)
            self.protodigis3 = nn.Parameter(torch.rand((5, self.numProtos, out_dim_caps)), requires_grad=True)
            self.protodigis4 = nn.Parameter(torch.rand((5, self.numProtos, out_dim_caps)), requires_grad=True)
            self.protodigis5 = nn.Parameter(torch.rand((2, self.numProtos, out_dim_caps)), requires_grad=True)
            self.protodigis_list = [self.protodigis0, self.protodigis1, self.protodigis2, self.protodigis3,
                                    self.protodigis4, self.protodigis5]
        # self.batchnorm

    def forwardCapsule(self, x_ex):
        capsout0 = self.attrOutLayer0(x_ex[:, 0, :])
        capsout1 = self.attrOutLayer1(x_ex[:, 1, :])
        capsout2 = self.attrOutLayer2(x_ex[:, 2, :])
        capsout3 = self.attrOutLayer3(x_ex[:, 3, :])
        capsout4 = self.attrOutLayer4(x_ex[:, 4, :])
        capsout5 = self.attrOutLayer5(x_ex[:, 5, :])
        if not self.args.dataset == "FunnyNodules":
            capsout6 = self.attrOutLayer6(x_ex[:, 6, :])
        if self.args.dataset in ["LIDC","Chexbert"]:
            capsout7 = self.attrOutLayer7(x_ex[:, 7, :])
            if self.args.dataset == "Chexbert":
                capsout8 = self.attrOutLayer8(x_ex[:, 8, :])
                capsout9 = self.attrOutLayer9(x_ex[:, 9, :])
                capsout10 = self.attrOutLayer10(x_ex[:, 10, :])
                capsout11 = self.attrOutLayer11(x_ex[:, 11, :])
                capsout12 = self.attrOutLayer12(x_ex[:, 12, :])
        if self.args.dataset == "LIDC":
            pred_attr = torch.cat((capsout0, capsout1, capsout2, capsout3, capsout4, capsout5, capsout6, capsout7), dim=1)
        elif self.args.dataset == "Chexbert":
            pred_attr = torch.cat((capsout0, capsout1, capsout2, capsout3, capsout4, capsout5, capsout6, capsout7,
                                   capsout8, capsout9,capsout10,capsout11,capsout12), dim=1)
        elif self.args.dataset == "derm7pt":
            pred_attr = torch.cat((capsout0, capsout1, capsout2, capsout3, capsout4, capsout5, capsout6), dim=1)
        elif self.args.dataset == "FunnyNodules":
            pred_attr = torch.cat((capsout0, capsout1, capsout2, capsout3, capsout4, capsout5), dim=1)

        reconstruction = self.decoder(x_ex)
        pred = self.predOutLayers(x_ex)

        return pred, pred_attr, reconstruction.view(-1, *self.input_size)

    def getdigitcaps(self,x):
        x = self.relu(self.conv1(x))
        x = self.primarycaps(x)
        x = self.digitcaps(x)
        return x

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.primarycaps(x)
        x = self.digitcaps(x)

        # attribute out
        capsout0 = self.attrOutLayer0(x[:, 0, :])
        capsout1 = self.attrOutLayer1(x[:, 1, :])
        capsout2 = self.attrOutLayer2(x[:, 2, :])
        capsout3 = self.attrOutLayer3(x[:, 3, :])
        capsout4 = self.attrOutLayer4(x[:, 4, :])
        capsout5 = self.attrOutLayer5(x[:, 5, :])
        if not self.args.dataset == "FunnyNodules":
            capsout6 = self.attrOutLayer6(x[:, 6, :])
        if self.args.dataset in ["LIDC", "Chexbert"]:
            capsout7 = self.attrOutLayer7(x[:, 7, :])
            if self.args.dataset == "Chexbert":
                capsout8 = self.attrOutLayer8(x[:, 8, :])
                capsout9 = self.attrOutLayer9(x[:, 9, :])
                capsout10 = self.attrOutLayer10(x[:, 10, :])
                capsout11 = self.attrOutLayer11(x[:, 11, :])
                capsout12 = self.attrOutLayer12(x[:, 12, :])

        if self.args.dataset == "LIDC":
            pred_attr = torch.cat((capsout0, capsout1, capsout2, capsout3, capsout4, capsout5, capsout6, capsout7), dim=1)
        elif self.args.dataset == "Chexbert":
            pred_attr = torch.cat((capsout0, capsout1, capsout2, capsout3, capsout4, capsout5, capsout6, capsout7,
                                   capsout8,capsout9,capsout10,capsout11,capsout12), dim=1)
        elif self.args.dataset == "derm7pt":
            pred_attr = torch.cat((capsout0, capsout1, capsout2, capsout3, capsout4, capsout5, capsout6), dim=1)
        elif self.args.dataset == "FunnyNodules":
            pred_attr = torch.cat((capsout0, capsout1, capsout2, capsout3, capsout4, capsout5), dim=1)

        # reconstruction
        reconstruction = self.decoder(x)

        # prediction out
        pred = self.predOutLayers(x)

        dists_to_protos = self.getDistance(x)
        return pred, pred_attr, reconstruction.view(-1, *self.input_size), dists_to_protos

    def forward_capsuleDIManalysis(self, x):
        x = self.relu(self.conv1(x))
        x = self.primarycaps(x)
        x = self.digitcaps(x)

        allattroutlayers = [self.attrOutLayer0, self.attrOutLayer1, self.attrOutLayer2, self.attrOutLayer3,
                            self.attrOutLayer4, self.attrOutLayer5, self.attrOutLayer6, self.attrOutLayer7]
        if self.args.dataset == "Chexbert":
            allattroutlayers = [self.attrOutLayer0, self.attrOutLayer1, self.attrOutLayer2, self.attrOutLayer3,
                                self.attrOutLayer4, self.attrOutLayer5, self.attrOutLayer6, self.attrOutLayer7,
                                self.attrOutLayer8, self.attrOutLayer9, self.attrOutLayer10, self.attrOutLayer11,
                                self.attrOutLayer12]

        # prediction out
        pred = self.predOutLayers(x)

        myquantile = 0.9

        matrix_pred_attr = np.zeros((x.shape[1], self.out_dim_caps, self.out_dim_caps))
        matrix_pred_attr_benign = np.zeros((x.shape[1], self.out_dim_caps, self.out_dim_caps))
        matrix_pred_attr_malignant = np.zeros((x.shape[1], self.out_dim_caps, self.out_dim_caps))
        pred_attr_withSecond_result = np.zeros((8))
        for attri in range(x.shape[1]):
            for sai in range(pred.shape[0]):
                attri_acti_all = torch.flatten(x[sai]) * self.predOutLayers[1].weight[torch.argmax(pred[sai])]
                quantile = torch.quantile(attri_acti_all, myquantile)
                most_q_important_cpsdimPRED = (attri_acti_all > quantile).nonzero()
                all_important_attri = []
                attricapsdim_array = torch.tensor(list(range(0, len(attri_acti_all) + 1, self.out_dim_caps))).to("cuda")
                for importantcapsdim in most_q_important_cpsdimPRED:
                    if not ((torch.argwhere(importantcapsdim[0] < attricapsdim_array)[0] - 1) in all_important_attri):
                        all_important_attri.append(
                            (torch.argwhere(importantcapsdim[0] < attricapsdim_array)[0] - 1).item())
                if attri in all_important_attri:
                    importantattriDIMforpred = torch.argmax(
                        (torch.flatten(x[sai]) * self.predOutLayers[1].weight[torch.argmax(pred[sai])])[
                        attri * self.out_dim_caps:(attri + 1) * self.out_dim_caps]).item()
                    importantattriDIMforattri = torch.argmax(
                        (x[sai, attri] * allattroutlayers[attri][0].weight[0])).item()
                    matrix_pred_attr[attri, importantattriDIMforpred, importantattriDIMforattri] += 1
                    if torch.argmax(pred[sai]) < 2:
                        matrix_pred_attr_benign[attri, importantattriDIMforpred, importantattriDIMforattri] += 1
                    elif torch.argmax(pred[sai]) > 2:
                        matrix_pred_attr_malignant[attri, importantattriDIMforpred, importantattriDIMforattri] += 1

                    allactivofattriforpred = (torch.flatten(x[sai]) * self.predOutLayers[1].weight[
                        torch.argmax(pred[sai])])[
                                             attri * self.out_dim_caps:(attri + 1) * self.out_dim_caps]
                    importantattriDIMforpred_second = torch.argsort(allactivofattriforpred)[-2]
                    activ_arr = (x[sai, attri] * allattroutlayers[attri][0].weight[0])
                    importantattriDIMforattri_second = torch.argsort(activ_arr)[-2]
                    if importantattriDIMforpred == importantattriDIMforattri:
                        pred_attr_withSecond_result[attri] += 1
                    elif importantattriDIMforpred == importantattriDIMforattri_second:
                        pred_attr_withSecond_result[attri] += 1
                    elif importantattriDIMforpred_second == importantattriDIMforattri:
                        pred_attr_withSecond_result[attri] += 1
                    elif importantattriDIMforpred_second == importantattriDIMforattri_second:
                        pred_attr_withSecond_result[attri] += 1

        attricomparisonforpredimportance = np.zeros((8))
        attricomparisonforpredimportance_benign = np.zeros((8))
        attricomparisonforpredimportance_malignant = np.zeros((8))
        for sai in range(pred.shape[0]):
            attri_acti_all = torch.flatten(x[sai]) * self.predOutLayers[1].weight[torch.argmax(pred[sai])]
            quantile = torch.quantile(attri_acti_all, myquantile)
            most_q_important_cpsdimPRED = (attri_acti_all>quantile).nonzero()
            all_important_attri = []
            attricapsdim_array = torch.tensor(list(range(0,len(attri_acti_all)+1, self.out_dim_caps))).to("cuda")
            for importantcapsdim in most_q_important_cpsdimPRED:
                if not ((torch.argwhere(importantcapsdim[0]<attricapsdim_array)[0]-1) in all_important_attri):
                    all_important_attri.append((torch.argwhere(importantcapsdim[0]<attricapsdim_array)[0]-1).item())
            for important_attri in all_important_attri:
                attricomparisonforpredimportance[important_attri] += 1
                if torch.argmax(pred[sai]) < 2:
                    attricomparisonforpredimportance_benign[important_attri] += 1
                elif torch.argmax(pred[sai]) > 2:
                    attricomparisonforpredimportance_malignant[important_attri] += 1

        return attricomparisonforpredimportance, matrix_pred_attr, pred_attr_withSecond_result, attricomparisonforpredimportance_benign, attricomparisonforpredimportance_malignant, matrix_pred_attr_benign, matrix_pred_attr_malignant

    def getDistance(self, x):
        """
        Capsule wise calculation of distance to closest prototype vector
        :param x: vectors to calculate distance to
        :return: distances to closest protoype vector
        """

        xreshaped = torch.unsqueeze(x, dim=1)
        xreshaped = torch.unsqueeze(xreshaped, dim=1)
        protoreshaped_list = []
        for i in range(len(self.protodigis_list)):
            protoreshaped_list.append(torch.unsqueeze(self.protodigis_list[i], dim=0))
        dists_to_protos = []
        for i in range(len(self.protodigis_list)):
            dists_to_protos.append((xreshaped[:, :, :, i, :] - protoreshaped_list[i]).pow(2).sum(-1).sqrt())

        if self.args.dataset == "Chexbert":
            protoreshaped0 = torch.unsqueeze(self.protodigis0, dim=0)
            protoreshaped1 = torch.unsqueeze(self.protodigis1, dim=0)
            protoreshaped2 = torch.unsqueeze(self.protodigis2, dim=0)
            protoreshaped3 = torch.unsqueeze(self.protodigis3, dim=0)
            protoreshaped4 = torch.unsqueeze(self.protodigis4, dim=0)
            protoreshaped5 = torch.unsqueeze(self.protodigis5, dim=0)
            protoreshaped6 = torch.unsqueeze(self.protodigis6, dim=0)
            protoreshaped7 = torch.unsqueeze(self.protodigis7, dim=0)
            protoreshaped8 = torch.unsqueeze(self.protodigis8, dim=0)
            protoreshaped9 = torch.unsqueeze(self.protodigis9, dim=0)
            protoreshaped10 = torch.unsqueeze(self.protodigis10, dim=0)
            protoreshaped11 = torch.unsqueeze(self.protodigis11, dim=0)
            protoreshaped12 = torch.unsqueeze(self.protodigis12, dim=0)
            dists_0 = (xreshaped[:, :, :, 0, :] - protoreshaped0).pow(2).sum(-1).sqrt()
            dists_1 = (xreshaped[:, :, :, 1, :] - protoreshaped1).pow(2).sum(-1).sqrt()
            dists_2 = (xreshaped[:, :, :, 2, :] - protoreshaped2).pow(2).sum(-1).sqrt()
            dists_3 = (xreshaped[:, :, :, 3, :] - protoreshaped3).pow(2).sum(-1).sqrt()
            dists_4 = (xreshaped[:, :, :, 4, :] - protoreshaped4).pow(2).sum(-1).sqrt()
            dists_5 = (xreshaped[:, :, :, 5, :] - protoreshaped5).pow(2).sum(-1).sqrt()
            dists_6 = (xreshaped[:, :, :, 6, :] - protoreshaped6).pow(2).sum(-1).sqrt()
            dists_7 = (xreshaped[:, :, :, 7, :] - protoreshaped7).pow(2).sum(-1).sqrt()
            dists_8 = (xreshaped[:, :, :, 8, :] - protoreshaped8).pow(2).sum(-1).sqrt()
            dists_9 = (xreshaped[:, :, :, 9, :] - protoreshaped9).pow(2).sum(-1).sqrt()
            dists_10 = (xreshaped[:, :, :, 10, :] - protoreshaped10).pow(2).sum(-1).sqrt()
            dists_11 = (xreshaped[:, :, :, 11, :] - protoreshaped11).pow(2).sum(-1).sqrt()
            dists_12 = (xreshaped[:, :, :, 12, :] - protoreshaped12).pow(2).sum(-1).sqrt()
            dists_to_protos = [dists_0, dists_1, dists_2, dists_3, dists_4, dists_5, dists_6, dists_7,
                               dists_8, dists_9, dists_10, dists_11, dists_12]


        return dists_to_protos
