"""
Push prototypes: update prototype vectors with samples of data_loader and save respective original image.

Author: Luisa Gallée, Github: `https://github.com/XRad-Ulm/HierViT`
"""

import torch

def pushprotos(model_push, data_loader, idx_with_attri, args):
    model_push.eval()
    if args.dataset == "LIDC":
        mindists_allcpsi = [torch.ones((model_push.protodigis0.shape[0], model_push.protodigis0.shape[1])) * torch.inf,
                            torch.ones((model_push.protodigis1.shape[0], model_push.protodigis1.shape[1])) * torch.inf,
                            torch.ones((model_push.protodigis2.shape[0], model_push.protodigis2.shape[1])) * torch.inf,
                            torch.ones((model_push.protodigis3.shape[0], model_push.protodigis3.shape[1])) * torch.inf,
                            torch.ones((model_push.protodigis4.shape[0], model_push.protodigis4.shape[1])) * torch.inf,
                            torch.ones((model_push.protodigis5.shape[0], model_push.protodigis5.shape[1])) * torch.inf,
                            torch.ones((model_push.protodigis6.shape[0], model_push.protodigis6.shape[1])) * torch.inf,
                            torch.ones((model_push.protodigis7.shape[0], model_push.protodigis7.shape[1])) * torch.inf]
    elif args.dataset == "FunnyNodules":
        mindists_allcpsi = [torch.ones((model_push.protodigis0.shape[0], model_push.protodigis0.shape[1])) * torch.inf,
                            torch.ones((model_push.protodigis1.shape[0], model_push.protodigis1.shape[1])) * torch.inf,
                            torch.ones((model_push.protodigis2.shape[0], model_push.protodigis2.shape[1])) * torch.inf,
                            torch.ones((model_push.protodigis3.shape[0], model_push.protodigis3.shape[1])) * torch.inf,
                            torch.ones((model_push.protodigis4.shape[0], model_push.protodigis4.shape[1])) * torch.inf,
                            torch.ones((model_push.protodigis5.shape[0], model_push.protodigis5.shape[1])) * torch.inf]
    elif args.dataset == "derm7pt":
        mindists_allcpsi = [torch.ones((model_push.protodigis0.shape[0], model_push.protodigis0.shape[1])) * torch.inf,
                            torch.ones((model_push.protodigis1.shape[0], model_push.protodigis1.shape[1])) * torch.inf,
                            torch.ones((model_push.protodigis2.shape[0], model_push.protodigis2.shape[1])) * torch.inf,
                            torch.ones((model_push.protodigis3.shape[0], model_push.protodigis3.shape[1])) * torch.inf,
                            torch.ones((model_push.protodigis4.shape[0], model_push.protodigis4.shape[1])) * torch.inf,
                            torch.ones((model_push.protodigis5.shape[0], model_push.protodigis5.shape[1])) * torch.inf,
                            torch.ones((model_push.protodigis6.shape[0], model_push.protodigis6.shape[1])) * torch.inf]
    elif args.dataset == "Chexbert":
        mindists_allcpsi = [torch.ones((model_push.protodigis0.shape[0], model_push.protodigis0.shape[1])) * torch.inf,
                            torch.ones((model_push.protodigis1.shape[0], model_push.protodigis1.shape[1])) * torch.inf,
                            torch.ones((model_push.protodigis2.shape[0], model_push.protodigis2.shape[1])) * torch.inf,
                            torch.ones((model_push.protodigis3.shape[0], model_push.protodigis3.shape[1])) * torch.inf,
                            torch.ones((model_push.protodigis4.shape[0], model_push.protodigis4.shape[1])) * torch.inf,
                            torch.ones((model_push.protodigis5.shape[0], model_push.protodigis5.shape[1])) * torch.inf,
                            torch.ones((model_push.protodigis6.shape[0], model_push.protodigis6.shape[1])) * torch.inf,
                            torch.ones((model_push.protodigis7.shape[0], model_push.protodigis7.shape[1])) * torch.inf,
                            torch.ones((model_push.protodigis8.shape[0], model_push.protodigis8.shape[1])) * torch.inf,
                            torch.ones((model_push.protodigis9.shape[0], model_push.protodigis9.shape[1])) * torch.inf,
                            torch.ones((model_push.protodigis10.shape[0], model_push.protodigis10.shape[1])) * torch.inf,
                            torch.ones((model_push.protodigis11.shape[0], model_push.protodigis11.shape[1])) * torch.inf,
                            torch.ones((model_push.protodigis12.shape[0], model_push.protodigis12.shape[1])) * torch.inf]
    if args.dataset in ["LIDC", "derm7pt","FunnyNodules"]:
        if model_push.threeD:
            mindists_X0 = torch.zeros(
                (model_push.protodigis0.shape[0], model_push.protodigis0.shape[1], model_push.input_size[0],
                 model_push.input_size[1], model_push.input_size[2], model_push.input_size[3]))
            mindists_X1 = torch.zeros(
                (model_push.protodigis1.shape[0], model_push.protodigis1.shape[1], model_push.input_size[0],
                 model_push.input_size[1], model_push.input_size[2], model_push.input_size[3]))
            mindists_X2 = torch.zeros(
                (model_push.protodigis2.shape[0], model_push.protodigis2.shape[1], model_push.input_size[0],
                 model_push.input_size[1], model_push.input_size[2], model_push.input_size[3]))
            mindists_X3 = torch.zeros(
                (model_push.protodigis3.shape[0], model_push.protodigis3.shape[1], model_push.input_size[0],
                 model_push.input_size[1], model_push.input_size[2], model_push.input_size[3]))
            mindists_X4 = torch.zeros(
                (model_push.protodigis4.shape[0], model_push.protodigis4.shape[1], model_push.input_size[0],
                 model_push.input_size[1], model_push.input_size[2], model_push.input_size[3]))
            mindists_X5 = torch.zeros(
                (model_push.protodigis5.shape[0], model_push.protodigis5.shape[1], model_push.input_size[0],
                 model_push.input_size[1], model_push.input_size[2], model_push.input_size[3]))
            mindists_X6 = torch.zeros(
                (model_push.protodigis6.shape[0], model_push.protodigis6.shape[1], model_push.input_size[0],
                 model_push.input_size[1], model_push.input_size[2], model_push.input_size[3]))
            mindists_X7 = torch.zeros(
                (model_push.protodigis7.shape[0], model_push.protodigis7.shape[1], model_push.input_size[0],
                 model_push.input_size[1], model_push.input_size[2], model_push.input_size[3]))
        else:
            if args.dataset == "derm7pt":
                mindists_X0 = torch.zeros(
                    (model_push.protodigis0.shape[0], model_push.protodigis0.shape[1], model_push.input_size[0],
                     model_push.input_size[1], model_push.input_size[2]))
                mindists_X1 = torch.zeros(
                    (model_push.protodigis1.shape[0], model_push.protodigis1.shape[1], model_push.input_size[0],
                     model_push.input_size[1], model_push.input_size[2]))
                mindists_X2 = torch.zeros(
                    (model_push.protodigis2.shape[0], model_push.protodigis2.shape[1], model_push.input_size[0],
                     model_push.input_size[1], model_push.input_size[2]))
                mindists_X3 = torch.zeros(
                    (model_push.protodigis3.shape[0], model_push.protodigis3.shape[1], model_push.input_size[0],
                     model_push.input_size[1], model_push.input_size[2]))
                mindists_X4 = torch.zeros(
                    (model_push.protodigis4.shape[0], model_push.protodigis4.shape[1], model_push.input_size[0],
                     model_push.input_size[1], model_push.input_size[2]))
                mindists_X5 = torch.zeros(
                    (model_push.protodigis5.shape[0], model_push.protodigis5.shape[1], model_push.input_size[0],
                     model_push.input_size[1], model_push.input_size[2]))
                mindists_X6 = torch.zeros(
                    (model_push.protodigis6.shape[0], model_push.protodigis6.shape[1], model_push.input_size[0],
                     model_push.input_size[1], model_push.input_size[2]))
            if args.dataset == "LIDC":
                mindists_X0 = torch.zeros(
                    (model_push.protodigis0.shape[0], model_push.protodigis0.shape[1], model_push.input_size[0],
                     model_push.input_size[1], model_push.input_size[2]))
                mindists_X1 = torch.zeros(
                    (model_push.protodigis1.shape[0], model_push.protodigis1.shape[1], model_push.input_size[0],
                     model_push.input_size[1], model_push.input_size[2]))
                mindists_X2 = torch.zeros(
                    (model_push.protodigis2.shape[0], model_push.protodigis2.shape[1], model_push.input_size[0],
                     model_push.input_size[1], model_push.input_size[2]))
                mindists_X3 = torch.zeros(
                    (model_push.protodigis3.shape[0], model_push.protodigis3.shape[1], model_push.input_size[0],
                     model_push.input_size[1], model_push.input_size[2]))
                mindists_X4 = torch.zeros(
                    (model_push.protodigis4.shape[0], model_push.protodigis4.shape[1], model_push.input_size[0],
                     model_push.input_size[1], model_push.input_size[2]))
                mindists_X5 = torch.zeros(
                    (model_push.protodigis5.shape[0], model_push.protodigis5.shape[1], model_push.input_size[0],
                     model_push.input_size[1], model_push.input_size[2]))
                mindists_X6 = torch.zeros(
                    (model_push.protodigis6.shape[0], model_push.protodigis6.shape[1], model_push.input_size[0],
                     model_push.input_size[1], model_push.input_size[2]))
                mindists_X7 = torch.zeros(
                    (model_push.protodigis7.shape[0], model_push.protodigis7.shape[1], model_push.input_size[0],
                     model_push.input_size[1], model_push.input_size[2]))
            if args.dataset == "FunnyNodules":
                mindists_X0 = torch.zeros(
                    (model_push.protodigis0.shape[0], model_push.protodigis0.shape[1], model_push.input_size[0],
                     model_push.input_size[1], model_push.input_size[2]))
                mindists_X1 = torch.zeros(
                    (model_push.protodigis1.shape[0], model_push.protodigis1.shape[1], model_push.input_size[0],
                     model_push.input_size[1], model_push.input_size[2]))
                mindists_X2 = torch.zeros(
                    (model_push.protodigis2.shape[0], model_push.protodigis2.shape[1], model_push.input_size[0],
                     model_push.input_size[1], model_push.input_size[2]))
                mindists_X3 = torch.zeros(
                    (model_push.protodigis3.shape[0], model_push.protodigis3.shape[1], model_push.input_size[0],
                     model_push.input_size[1], model_push.input_size[2]))
                mindists_X4 = torch.zeros(
                    (model_push.protodigis4.shape[0], model_push.protodigis4.shape[1], model_push.input_size[0],
                     model_push.input_size[1], model_push.input_size[2]))
                mindists_X5 = torch.zeros(
                    (model_push.protodigis5.shape[0], model_push.protodigis5.shape[1], model_push.input_size[0],
                     model_push.input_size[1], model_push.input_size[2]))

        if args.dataset == "LIDC":
            mindists_X_allcpsi = [mindists_X0, mindists_X1, mindists_X2, mindists_X3, mindists_X4, mindists_X5, mindists_X6,
                                  mindists_X7]
        if args.dataset == "derm7pt":
            mindists_X_allcpsi = [mindists_X0, mindists_X1, mindists_X2, mindists_X3, mindists_X4, mindists_X5, mindists_X6]
        if args.dataset == "FunnyNodules":
            mindists_X_allcpsi = [mindists_X0, mindists_X1, mindists_X2, mindists_X3, mindists_X4, mindists_X5]
        if args.dataset == "LIDC":
            protos_total_id = [torch.ones((model_push.protodigis0.shape[0], model_push.protodigis0.shape[1], 4))-2,
                               torch.ones((model_push.protodigis1.shape[0], model_push.protodigis1.shape[1], 4))-2,
                               torch.ones((model_push.protodigis2.shape[0], model_push.protodigis2.shape[1], 4))-2,
                               torch.ones((model_push.protodigis3.shape[0], model_push.protodigis3.shape[1], 4))-2,
                               torch.ones((model_push.protodigis4.shape[0], model_push.protodigis4.shape[1], 4))-2,
                               torch.ones((model_push.protodigis5.shape[0], model_push.protodigis5.shape[1], 4))-2,
                               torch.ones((model_push.protodigis6.shape[0], model_push.protodigis6.shape[1], 4))-2,
                               torch.ones((model_push.protodigis7.shape[0], model_push.protodigis7.shape[1], 4))-2]
            mindists_sampledigis_allcps = [torch.zeros_like(model_push.protodigis0), torch.zeros_like(model_push.protodigis1),
                                           torch.zeros_like(model_push.protodigis2), torch.zeros_like(model_push.protodigis3),
                                           torch.zeros_like(model_push.protodigis4), torch.zeros_like(model_push.protodigis5),
                                           torch.zeros_like(model_push.protodigis6), torch.zeros_like(model_push.protodigis7)]
            mindists_alllabels = [torch.zeros((model_push.protodigis0.shape[0], model_push.protodigis0.shape[1], 9)),
                                        torch.zeros((model_push.protodigis1.shape[0], model_push.protodigis1.shape[1], 9)),
                                        torch.zeros((model_push.protodigis2.shape[0], model_push.protodigis2.shape[1], 9)),
                                        torch.zeros((model_push.protodigis3.shape[0], model_push.protodigis3.shape[1], 9)),
                                        torch.zeros((model_push.protodigis4.shape[0], model_push.protodigis4.shape[1], 9)),
                                        torch.zeros((model_push.protodigis5.shape[0], model_push.protodigis5.shape[1], 9)),
                                        torch.zeros((model_push.protodigis6.shape[0], model_push.protodigis6.shape[1], 9)),
                                        torch.zeros((model_push.protodigis7.shape[0], model_push.protodigis7.shape[1], 9))]
        if args.dataset == "derm7pt":
            protos_total_id = [
                torch.ones((model_push.protodigis0.shape[0], model_push.protodigis0.shape[1], 4)) - 2,
                torch.ones((model_push.protodigis1.shape[0], model_push.protodigis1.shape[1], 4)) - 2,
                torch.ones((model_push.protodigis2.shape[0], model_push.protodigis2.shape[1], 4)) - 2,
                torch.ones((model_push.protodigis3.shape[0], model_push.protodigis3.shape[1], 4)) - 2,
                torch.ones((model_push.protodigis4.shape[0], model_push.protodigis4.shape[1], 4)) - 2,
                torch.ones((model_push.protodigis5.shape[0], model_push.protodigis5.shape[1], 4)) - 2,
                torch.ones((model_push.protodigis6.shape[0], model_push.protodigis6.shape[1], 4)) - 2]
            mindists_sampledigis_allcps = [torch.zeros_like(model_push.protodigis0),
                                           torch.zeros_like(model_push.protodigis1),
                                           torch.zeros_like(model_push.protodigis2),
                                           torch.zeros_like(model_push.protodigis3),
                                           torch.zeros_like(model_push.protodigis4),
                                           torch.zeros_like(model_push.protodigis5),
                                           torch.zeros_like(model_push.protodigis6)]
            mindists_alllabels = [
                torch.zeros((model_push.protodigis0.shape[0], model_push.protodigis0.shape[1], 8)),
                torch.zeros((model_push.protodigis1.shape[0], model_push.protodigis1.shape[1], 8)),
                torch.zeros((model_push.protodigis2.shape[0], model_push.protodigis2.shape[1], 8)),
                torch.zeros((model_push.protodigis3.shape[0], model_push.protodigis3.shape[1], 8)),
                torch.zeros((model_push.protodigis4.shape[0], model_push.protodigis4.shape[1], 8)),
                torch.zeros((model_push.protodigis5.shape[0], model_push.protodigis5.shape[1], 8)),
                torch.zeros((model_push.protodigis6.shape[0], model_push.protodigis6.shape[1], 8))]

        if args.dataset == "FunnyNodules":
            protos_total_id = [
                torch.ones((model_push.protodigis0.shape[0], model_push.protodigis0.shape[1], 4)) - 2,
                torch.ones((model_push.protodigis1.shape[0], model_push.protodigis1.shape[1], 4)) - 2,
                torch.ones((model_push.protodigis2.shape[0], model_push.protodigis2.shape[1], 4)) - 2,
                torch.ones((model_push.protodigis3.shape[0], model_push.protodigis3.shape[1], 4)) - 2,
                torch.ones((model_push.protodigis4.shape[0], model_push.protodigis4.shape[1], 4)) - 2,
                torch.ones((model_push.protodigis5.shape[0], model_push.protodigis5.shape[1], 4)) - 2]
            mindists_sampledigis_allcps = [torch.zeros_like(model_push.protodigis0),
                                           torch.zeros_like(model_push.protodigis1),
                                           torch.zeros_like(model_push.protodigis2),
                                           torch.zeros_like(model_push.protodigis3),
                                           torch.zeros_like(model_push.protodigis4),
                                           torch.zeros_like(model_push.protodigis5)]
            mindists_alllabels = [
                torch.zeros((model_push.protodigis0.shape[0], model_push.protodigis0.shape[1], 7)),
                torch.zeros((model_push.protodigis1.shape[0], model_push.protodigis1.shape[1], 7)),
                torch.zeros((model_push.protodigis2.shape[0], model_push.protodigis2.shape[1], 7)),
                torch.zeros((model_push.protodigis3.shape[0], model_push.protodigis3.shape[1], 7)),
                torch.zeros((model_push.protodigis4.shape[0], model_push.protodigis4.shape[1], 7)),
                torch.zeros((model_push.protodigis5.shape[0], model_push.protodigis5.shape[1], 7))]

    elif args.dataset == "Chexbert":
        mindists_X0 = torch.zeros(
            (model_push.protodigis0.shape[0], model_push.protodigis0.shape[1], model_push.input_size[0],
             model_push.input_size[1], model_push.input_size[2]))
        mindists_X1 = torch.zeros(
            (model_push.protodigis1.shape[0], model_push.protodigis1.shape[1], model_push.input_size[0],
             model_push.input_size[1], model_push.input_size[2]))
        mindists_X2 = torch.zeros(
            (model_push.protodigis2.shape[0], model_push.protodigis2.shape[1], model_push.input_size[0],
             model_push.input_size[1], model_push.input_size[2]))
        mindists_X3 = torch.zeros(
            (model_push.protodigis3.shape[0], model_push.protodigis3.shape[1], model_push.input_size[0],
             model_push.input_size[1], model_push.input_size[2]))
        mindists_X4 = torch.zeros(
            (model_push.protodigis4.shape[0], model_push.protodigis4.shape[1], model_push.input_size[0],
             model_push.input_size[1], model_push.input_size[2]))
        mindists_X5 = torch.zeros(
            (model_push.protodigis5.shape[0], model_push.protodigis5.shape[1], model_push.input_size[0],
             model_push.input_size[1], model_push.input_size[2]))
        mindists_X6 = torch.zeros(
            (model_push.protodigis6.shape[0], model_push.protodigis6.shape[1], model_push.input_size[0],
             model_push.input_size[1], model_push.input_size[2]))
        mindists_X7 = torch.zeros(
            (model_push.protodigis7.shape[0], model_push.protodigis7.shape[1], model_push.input_size[0],
             model_push.input_size[1], model_push.input_size[2]))
        mindists_X8 = torch.zeros(
            (model_push.protodigis8.shape[0], model_push.protodigis8.shape[1], model_push.input_size[0],
             model_push.input_size[1], model_push.input_size[2]))
        mindists_X9 = torch.zeros(
            (model_push.protodigis9.shape[0], model_push.protodigis9.shape[1], model_push.input_size[0],
             model_push.input_size[1], model_push.input_size[2]))
        mindists_X10 = torch.zeros(
            (model_push.protodigis10.shape[0], model_push.protodigis10.shape[1], model_push.input_size[0],
             model_push.input_size[1], model_push.input_size[2]))
        mindists_X11 = torch.zeros(
            (model_push.protodigis11.shape[0], model_push.protodigis11.shape[1], model_push.input_size[0],
             model_push.input_size[1], model_push.input_size[2]))
        mindists_X12 = torch.zeros(
            (model_push.protodigis12.shape[0], model_push.protodigis12.shape[1], model_push.input_size[0],
             model_push.input_size[1], model_push.input_size[2]))
        mindists_X_allcpsi = [mindists_X0, mindists_X1, mindists_X2, mindists_X3, mindists_X4, mindists_X5, mindists_X6,
                              mindists_X7, mindists_X8, mindists_X9, mindists_X10, mindists_X11, mindists_X12]
        mindists_sampledigis_allcps = [torch.zeros_like(model_push.protodigis0), torch.zeros_like(model_push.protodigis1),
                                       torch.zeros_like(model_push.protodigis2), torch.zeros_like(model_push.protodigis3),
                                       torch.zeros_like(model_push.protodigis4), torch.zeros_like(model_push.protodigis5),
                                       torch.zeros_like(model_push.protodigis6), torch.zeros_like(model_push.protodigis7),
                                       torch.zeros_like(model_push.protodigis8), torch.zeros_like(model_push.protodigis9),
                                       torch.zeros_like(model_push.protodigis10), torch.zeros_like(model_push.protodigis11),
                                       torch.zeros_like(model_push.protodigis12)]
        mindists_alllabels = [torch.zeros((model_push.protodigis0.shape[0], model_push.protodigis0.shape[1], 14)),
                              torch.zeros((model_push.protodigis1.shape[0], model_push.protodigis1.shape[1], 14)),
                              torch.zeros((model_push.protodigis2.shape[0], model_push.protodigis2.shape[1], 14)),
                              torch.zeros((model_push.protodigis3.shape[0], model_push.protodigis3.shape[1], 14)),
                              torch.zeros((model_push.protodigis4.shape[0], model_push.protodigis4.shape[1], 14)),
                              torch.zeros((model_push.protodigis5.shape[0], model_push.protodigis5.shape[1], 14)),
                              torch.zeros((model_push.protodigis6.shape[0], model_push.protodigis6.shape[1], 14)),
                              torch.zeros((model_push.protodigis7.shape[0], model_push.protodigis7.shape[1], 14)),
                              torch.zeros((model_push.protodigis8.shape[0], model_push.protodigis8.shape[1], 14)),
                              torch.zeros((model_push.protodigis9.shape[0], model_push.protodigis9.shape[1], 14)),
                              torch.zeros((model_push.protodigis10.shape[0], model_push.protodigis10.shape[1], 14)),
                              torch.zeros((model_push.protodigis11.shape[0], model_push.protodigis11.shape[1], 14)),
                              torch.zeros((model_push.protodigis12.shape[0], model_push.protodigis12.shape[1], 14))]
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            print("Push: "+str(i) + " / " + str(len(data_loader)))
            if args.dataset == "LIDC":
                (x, y_mask, y_attributes, y_mal, sampleID, img_total) = data
                x, y_mask, y_attributes, y_mal = x.to("cuda", dtype=torch.float), y_mask.to("cuda", dtype=torch.float), \
                    y_attributes.to("cuda", dtype=torch.float), y_mal.to("cuda", dtype=torch.float)
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
                x, y_mask, y_attributes, y_mal = x.to("cuda", dtype=torch.float), y_mask.to("cuda", dtype=torch.float), \
                    y_attributes.to("cuda", dtype=torch.float), y_mal.to("cuda", dtype=torch.float)

            if args.base_model == "ViT":
                pred_outs, pred_recon = model_push(x)
                if args.dataset == "LIDC":
                    pred_attr = pred_outs[:,:8]
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
                dists_to_protos = model_push.getDistance(x)
            else:
                _, pred_attr, _, dists_to_protos = model_push(x)
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

            if args.dataset == "LIDC":
                attr_classes = [5, 4, 6, 5, 5, 5, 5, 5]
                _, max_y_mal = torch.max(y_mal, dim=-1)
            elif args.dataset == "Chexbert":
                max_y_mal = y_mal
                attr_classes = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
            elif args.dataset == "derm7pt":
                max_y_mal = y_mal
                attr_classes = [3,2,3,3,3,3,2]
            elif args.dataset == "FunnyNodules":
                max_y_mal = y_mal
                attr_classes = [5,5,5,5,5,2]

            for sai in range(x.shape[0]):
                if sampleID[sai] in idx_with_attri:
                    for capsule_idx in range(len(attr_classes)):
                        attrcorrectlypredicted=False
                        if args.dataset == "LIDC":
                            if torch.abs(y_attributes[sai, capsule_idx] - pred_attr[sai, capsule_idx]) < 0.25:
                                attrcorrectlypredicted=True
                        elif args.dataset == "FunnyNodules":
                            if torch.abs(y_attributes[sai, capsule_idx] - pred_attr[sai, capsule_idx]) < 0.25:
                                attrcorrectlypredicted=True
                        elif args.dataset == "Chexbert":
                            if torch.round(y_attributes[sai, capsule_idx]) == torch.round(pred_attr[sai, capsule_idx]):
                                attrcorrectlypredicted = True
                        elif args.dataset == "derm7pt":
                            if y_attributes[sai,capsule_idx] == torch.argmax(pred_attr[capsule_idx][sai],dim=-1):
                            #3,2,3,3,3,3,2
                                for protoidx1 in range(dists_to_protos[capsule_idx].shape[1]):
                                    if protoidx1 == y_attributes[sai,capsule_idx]:
                                        for protoidx2 in range(dists_to_protos[capsule_idx].shape[2]):
                                            if dists_to_protos[capsule_idx][sai, protoidx1, protoidx2] < \
                                                    mindists_allcpsi[capsule_idx][protoidx1, protoidx2]:
                                                mindists_allcpsi[capsule_idx][protoidx1, protoidx2] = \
                                                    dists_to_protos[capsule_idx][sai, protoidx1, protoidx2]
                                                mindists_X_allcpsi[capsule_idx][protoidx1, protoidx2] = x[sai].cpu()
                                                mindists_alllabels[capsule_idx][
                                                    protoidx1, protoidx2] = torch.cat(
                                                    (y_attributes[sai], max_y_mal[sai].unsqueeze(0)), 0)
                                                mindists_sampledigis_allcps[capsule_idx][protoidx1, protoidx2] = \
                                                    model_push.protodigis_list[capsule_idx][
                                                    protoidx1, protoidx2, :].cpu()


                        if attrcorrectlypredicted:
                            for protoidx in range(dists_to_protos[capsule_idx].shape[1]):
                                if args.dataset == "FunnyNodules":
                                    if capsule_idx < 5:
                                        correctprotoattr = False
                                        if (((y_attributes[sai, capsule_idx] < 0.125) and protoidx == 0) or
                                                ((0.125 <= y_attributes[sai, capsule_idx] < 0.375) and protoidx == 1) or
                                                ((0.375 <= y_attributes[sai, capsule_idx] < 0.625) and protoidx == 2) or
                                                ((0.625 <= y_attributes[sai, capsule_idx] < 0.875) and protoidx == 3) or
                                                ((y_attributes[sai, capsule_idx] >= 0.875) and protoidx == 4)):
                                            correctprotoattr = True
                                        if correctprotoattr:
                                            for protoidx2 in range(dists_to_protos[capsule_idx].shape[2]):
                                                if dists_to_protos[capsule_idx][sai, protoidx, protoidx2] < \
                                                        mindists_allcpsi[capsule_idx][
                                                            protoidx, protoidx2]:
                                                    mindists_allcpsi[capsule_idx][protoidx, protoidx2] = \
                                                        dists_to_protos[capsule_idx][
                                                            sai, protoidx, protoidx2]
                                                    if args.threeD:
                                                        mindists_X_allcpsi[capsule_idx][protoidx, protoidx2] = x[sai].permute(0,2,3,1).cpu()
                                                    else:
                                                        mindists_X_allcpsi[capsule_idx][protoidx, protoidx2] = x[sai].cpu()
                                                    mindists_alllabels[capsule_idx][
                                                        protoidx, protoidx2] = torch.cat(
                                                        (y_attributes[sai], max_y_mal[sai].unsqueeze(0)), 0)
                                                    mindists_sampledigis_allcps[capsule_idx][protoidx, protoidx2] = \
                                                        model_push.protodigis_list[capsule_idx][
                                                        protoidx, protoidx2, :].cpu()
                                    else:
                                        correctprotoattr = False
                                        if (((y_attributes[sai, capsule_idx] < 0.5) and protoidx == 0) or
                                                ((y_attributes[sai, capsule_idx] >= 0.5) and protoidx == 1)):
                                            correctprotoattr = True
                                        if correctprotoattr:
                                            for protoidx2 in range(dists_to_protos[capsule_idx].shape[2]):
                                                if dists_to_protos[capsule_idx][sai, protoidx, protoidx2] < \
                                                        mindists_allcpsi[capsule_idx][
                                                            protoidx, protoidx2]:
                                                    mindists_allcpsi[capsule_idx][protoidx, protoidx2] = \
                                                        dists_to_protos[capsule_idx][
                                                            sai, protoidx, protoidx2]
                                                    if args.threeD:
                                                        mindists_X_allcpsi[capsule_idx][protoidx, protoidx2] = x[sai].permute(0,2,3,1).cpu()
                                                    else:
                                                        mindists_X_allcpsi[capsule_idx][protoidx, protoidx2] = x[sai].cpu()
                                                    mindists_alllabels[capsule_idx][
                                                        protoidx, protoidx2] = torch.cat(
                                                        (y_attributes[sai], max_y_mal[sai].unsqueeze(0)), 0)
                                                    mindists_sampledigis_allcps[capsule_idx][protoidx, protoidx2] = \
                                                        model_push.protodigis_list[capsule_idx][
                                                        protoidx, protoidx2, :].cpu()
                                if args.dataset == "LIDC":
                                    if capsule_idx in [0, 3, 4, 5, 6, 7]:
                                        correctprotoattr=False
                                        if (((y_attributes[sai, capsule_idx] < 0.125) and protoidx == 0) or
                                                ((0.125 <= y_attributes[sai, capsule_idx] < 0.375) and protoidx == 1) or
                                                ((0.375 <= y_attributes[sai, capsule_idx] < 0.625) and protoidx == 2) or
                                                ((0.625 <= y_attributes[sai, capsule_idx] < 0.875) and protoidx == 3) or
                                                ((y_attributes[sai, capsule_idx] >= 0.875) and protoidx == 4)):
                                            correctprotoattr = True
                                        if correctprotoattr:
                                            for protoidx2 in range(dists_to_protos[capsule_idx].shape[2]):
                                                if dists_to_protos[capsule_idx][sai, protoidx, protoidx2] < \
                                                        mindists_allcpsi[capsule_idx][
                                                            protoidx, protoidx2]:
                                                    mindists_allcpsi[capsule_idx][protoidx, protoidx2] = \
                                                        dists_to_protos[capsule_idx][
                                                            sai, protoidx, protoidx2]
                                                    if args.threeD:
                                                        mindists_X_allcpsi[capsule_idx][protoidx, protoidx2] = x[sai].permute(0,2,3,1).cpu()
                                                    else:
                                                        mindists_X_allcpsi[capsule_idx][protoidx, protoidx2] = x[sai].cpu()
                                                        ti_split = img_total[sai].split('_')
                                                        for ti in range(4):
                                                            protos_total_id[capsule_idx][protoidx, protoidx2, ti] = int(ti_split[ti])
                                                    mindists_alllabels[capsule_idx][
                                                        protoidx, protoidx2] = torch.cat(
                                                        (y_attributes[sai], max_y_mal[sai].unsqueeze(0)), 0)
                                                    mindists_sampledigis_allcps[capsule_idx][protoidx, protoidx2] = \
                                                        model_push.protodigis_list[capsule_idx][
                                                        protoidx, protoidx2, :].cpu()
                                    elif capsule_idx == 1:
                                        correctprotoattr = False
                                        if (((y_attributes[sai, capsule_idx] < 0.16) and protoidx == 0) or
                                                ((0.16 <= y_attributes[sai, capsule_idx] < 0.49) and protoidx == 1) or
                                                ((0.49 <= y_attributes[sai, capsule_idx] < 0.82) and protoidx == 2) or
                                                ((y_attributes[sai, capsule_idx] >= 0.82) and protoidx == 3)):
                                            correctprotoattr = True
                                        if correctprotoattr:
                                            for protoidx2 in range(dists_to_protos[capsule_idx].shape[2]):
                                                if dists_to_protos[capsule_idx][sai, protoidx, protoidx2] < \
                                                        mindists_allcpsi[capsule_idx][
                                                            protoidx, protoidx2]:
                                                    mindists_allcpsi[capsule_idx][protoidx, protoidx2] = \
                                                        dists_to_protos[capsule_idx][
                                                            sai, protoidx, protoidx2]
                                                    if args.threeD:
                                                        mindists_X_allcpsi[capsule_idx][protoidx, protoidx2] = x[sai].permute(0, 2,3,1).cpu()
                                                    else:
                                                        mindists_X_allcpsi[capsule_idx][protoidx, protoidx2] = x[sai].cpu()
                                                        ti_split = img_total[sai].split('_')
                                                        for ti in range(4):
                                                            protos_total_id[capsule_idx][protoidx, protoidx2, ti] = \
                                                            int(ti_split[ti])
                                                    mindists_alllabels[capsule_idx][
                                                        protoidx, protoidx2] = torch.cat(
                                                        (y_attributes[sai], max_y_mal[sai].unsqueeze(0)), 0)
                                                    mindists_sampledigis_allcps[capsule_idx][protoidx, protoidx2] = \
                                                        model_push.protodigis_list[capsule_idx][
                                                        protoidx, protoidx2, :].cpu()
                                    elif capsule_idx == 2:
                                        correctprotoattr = False
                                        if (((y_attributes[sai, capsule_idx] < 0.1) and protoidx == 0) or
                                                ((0.1 <= y_attributes[sai, capsule_idx] < 0.3) and protoidx == 1) or
                                                ((0.3 <= y_attributes[sai, capsule_idx] < 0.5) and protoidx == 2) or
                                                ((0.5 <= y_attributes[sai, capsule_idx] < 0.7) and protoidx == 3) or
                                                ((0.7 <= y_attributes[sai, capsule_idx] < 0.9) and protoidx == 4) or
                                                ((y_attributes[sai, capsule_idx] >= 0.9) and protoidx == 5)):
                                            correctprotoattr = True
                                        if correctprotoattr:
                                            for protoidx2 in range(dists_to_protos[capsule_idx].shape[2]):
                                                if dists_to_protos[capsule_idx][sai, protoidx, protoidx2] < \
                                                        mindists_allcpsi[capsule_idx][
                                                            protoidx, protoidx2]:
                                                    mindists_allcpsi[capsule_idx][protoidx, protoidx2] = \
                                                        dists_to_protos[capsule_idx][
                                                            sai, protoidx, protoidx2]
                                                    if args.threeD:
                                                        mindists_X_allcpsi[capsule_idx][protoidx, protoidx2] = x[sai].permute(0, 2,3,1).cpu()
                                                    else:
                                                        mindists_X_allcpsi[capsule_idx][protoidx, protoidx2] = x[sai].cpu()
                                                        ti_split = img_total[sai].split('_')
                                                        for ti in range(4):
                                                            protos_total_id[capsule_idx][protoidx, protoidx2, ti] = int(ti_split[ti])
                                                    mindists_alllabels[capsule_idx][
                                                        protoidx, protoidx2] = torch.cat(
                                                        (y_attributes[sai], max_y_mal[sai].unsqueeze(0)), 0)
                                                    mindists_sampledigis_allcps[capsule_idx][protoidx, protoidx2] = \
                                                        model_push.protodigis_list[capsule_idx][
                                                        protoidx, protoidx2, :].cpu()
                                elif args.dataset == "Chexbert":
                                    correctprotoattr = False
                                    if (((y_attributes[sai, capsule_idx] < 0.5) and protoidx == 0) or
                                            ((y_attributes[sai, capsule_idx] >= 0.5) and protoidx == 1)):
                                        correctprotoattr = True
                                    if correctprotoattr:
                                        for protoidx2 in range(dists_to_protos[capsule_idx].shape[2]):
                                            if dists_to_protos[capsule_idx][sai, protoidx, protoidx2] < \
                                                    mindists_allcpsi[capsule_idx][
                                                        protoidx, protoidx2]:
                                                mindists_allcpsi[capsule_idx][protoidx, protoidx2] = \
                                                    dists_to_protos[capsule_idx][
                                                        sai, protoidx, protoidx2]
                                                mindists_X_allcpsi[capsule_idx][protoidx, protoidx2] = x[sai].cpu()
                                                mindists_alllabels[capsule_idx][
                                                    protoidx, protoidx2] = torch.cat(
                                                    (y_attributes[sai], max_y_mal[sai].unsqueeze(0)), 0)
                                                mindists_sampledigis_allcps[capsule_idx][protoidx, protoidx2] = \
                                                    model_push.protodigis_list[capsule_idx][
                                                    protoidx, protoidx2, :].cpu()
    if args.dataset == "LIDC":
        model_push.protodigis0.data.copy_(mindists_sampledigis_allcps[0].cuda())
        model_push.protodigis1.data.copy_(mindists_sampledigis_allcps[1].cuda())
        model_push.protodigis2.data.copy_(mindists_sampledigis_allcps[2].cuda())
        model_push.protodigis3.data.copy_(mindists_sampledigis_allcps[3].cuda())
        model_push.protodigis4.data.copy_(mindists_sampledigis_allcps[4].cuda())
        model_push.protodigis5.data.copy_(mindists_sampledigis_allcps[5].cuda())
        model_push.protodigis6.data.copy_(mindists_sampledigis_allcps[6].cuda())
        model_push.protodigis7.data.copy_(mindists_sampledigis_allcps[7].cuda())
    if args.dataset == "FunnyNodules":
        model_push.protodigis0.data.copy_(mindists_sampledigis_allcps[0].cuda())
        model_push.protodigis1.data.copy_(mindists_sampledigis_allcps[1].cuda())
        model_push.protodigis2.data.copy_(mindists_sampledigis_allcps[2].cuda())
        model_push.protodigis3.data.copy_(mindists_sampledigis_allcps[3].cuda())
        model_push.protodigis4.data.copy_(mindists_sampledigis_allcps[4].cuda())
        model_push.protodigis5.data.copy_(mindists_sampledigis_allcps[5].cuda())
    if args.dataset == "derm7pt":
        model_push.protodigis0.data.copy_(mindists_sampledigis_allcps[0].cuda())
        model_push.protodigis1.data.copy_(mindists_sampledigis_allcps[1].cuda())
        model_push.protodigis2.data.copy_(mindists_sampledigis_allcps[2].cuda())
        model_push.protodigis3.data.copy_(mindists_sampledigis_allcps[3].cuda())
        model_push.protodigis4.data.copy_(mindists_sampledigis_allcps[4].cuda())
        model_push.protodigis5.data.copy_(mindists_sampledigis_allcps[5].cuda())
        model_push.protodigis6.data.copy_(mindists_sampledigis_allcps[6].cuda())
    if args.dataset == "Chexbert":
        model_push.protodigis8.data.copy_(mindists_sampledigis_allcps[8].cuda())
        model_push.protodigis9.data.copy_(mindists_sampledigis_allcps[9].cuda())
        model_push.protodigis10.data.copy_(mindists_sampledigis_allcps[10].cuda())
        model_push.protodigis11.data.copy_(mindists_sampledigis_allcps[11].cuda())
        model_push.protodigis12.data.copy_(mindists_sampledigis_allcps[12].cuda())

    return model_push, mindists_X_allcpsi, mindists_alllabels, protos_total_id
