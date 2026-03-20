"""
Pytorch implementation of HierViT

Author: Luisa Gallée, Github: `https://github.com/XRad-Ulm/HierViT`
"""

import sys

import matplotlib.pyplot as plt
import torch
import datetime
import numpy as np
from ProtoCaps import ProtoCapsNet
import wandb
from push import pushprotos
from train import train_model
# from data_loader import load_lidc,load_chexbert,load_derm7pt
from test import test, test_indepth, test_indepth_attripredCorr, test_getweightedsharedinfo, test_show_inference
from PIL import Image
from HierViT import CustomVisionTransformer

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="FunnyNodules")
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.02, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lam_recon', default=0.512, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--warmup', default=100, type=int,
                        help="Number of epochs before prototypes are fitted.")
    parser.add_argument('--push_step', default=10, type=int,
                        help="Prototypes are pushed every [push_step] epoch.")
    parser.add_argument('--split_number', default=0, type=int)
    parser.add_argument('--shareAttrLabels', default=1.0, type=float)
    parser.add_argument('--threeD', default=False, type=bool)
    parser.add_argument('--resize_shape', nargs='+', type=int, default=[32, 32],
                        help="Size of boxes cropped out of CT volumes as model input")
    parser.add_argument('--train', default=False, type=bool)
    parser.add_argument('--test', default=False, type=bool)
    parser.add_argument('--model_path', type=str, help="Set the path of the model to be tested")
    parser.add_argument('--epoch', type=int, help="Set the epoch of chosen model")
    # small out_dim_caps leads to different prototypes per attribute class
    parser.add_argument('--out_dim_caps', type=int, default=16, help="Set dimension of output capsule vectors.")
    parser.add_argument('--num_protos', type=int, default=16, help="Set number of prototypes per attribute class")
    parser.add_argument('--dataset', type=str, default="FunnyNodules", choices=["FunnyNodules", "LIDC", "Chexbert", "derm7pt"])
    parser.add_argument('--base_model', default="ViT", choices=["ConvNet", "ViT","ResNetMT","DenseNetMT"])
    parser.add_argument('--onlyTar', default=False, type=bool)

    args = parser.parse_args()
    print(args)

    if (not args.train and not args.test) or (args.train and args.test):
        raise TypeError(
            "Please specify, whether you want to run the training or testing code by setting the parameter --train=True or --test=True")

    if args.dataset == "FunnyNodules":
        from dataset_generator import create_dataloader
        train_loader = create_dataloader(num_samples=1800, batch_size=args.batch_size, img_size=args.resize_shape[0])
        val_loader = create_dataloader(num_samples=200, batch_size=args.batch_size, img_size=args.resize_shape[0])
        test_loader = create_dataloader(num_samples=500, batch_size=args.batch_size, img_size=args.resize_shape[0])
        numattributes = next(iter(train_loader))[2].shape[1]
        print("num attributes "+str(numattributes))
        print("training samples: " + str(len(train_loader.dataset)))
        print("val samples: " + str(len(val_loader.dataset)))
        print("test samples: " + str(len(test_loader.dataset)))
        for images, masks, attrs, target, sampleID in train_loader:
            print("Batch Bilder:", images.shape)
            print(torch.min(images))
            print(torch.max(images))
            print("Batch Masks:", masks.shape)
            print("Batch Attribut-Dict:", attrs)
            print("Batch Target:", target)
            print("Batch sampleID:", sampleID)
            break

        from matplotlib import rcParams
        rcParams['font.family'] = 'serif'
        rcParams['font.serif'] = ['Times New Roman']
        rcParams['font.size'] = 30  # base font size for labels, ticks
        rcParams['axes.titlesize'] = 32  # title size
        rcParams['axes.labelsize'] = 30  # axis label size
        rcParams['xtick.labelsize'] = 28  # x tick labels
        rcParams['ytick.labelsize'] = 28  # y tick labels
        all_y = []
        all_attris_names=["roundness", "spiculation", "edge sharpness","size", "intensity", "internal structure"]
        all_attris = {'0': [],'1': [],'2': [],'3': [],'4': [], '5': []}
        for data in test_loader:
            (x, y_mask, y_attributes, y_mal, sampleID) = data
            for ai in range(y_attributes.shape[-1]):
                if ai<5:
                    all_attris[str(ai)].extend((y_attributes[:,ai]*4+1).ravel().detach().cpu().numpy())
                else:
                    all_attris[str(ai)].extend((y_attributes[:,ai]).ravel().detach().cpu().numpy())
            all_y.extend((y_mal*4+1).ravel().detach().cpu().numpy())
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        bins=5
        axes[0].hist(all_y, bins=bins, edgecolor='black')
        axes[0].set_title('target')
        axes[0].set_xticks([1, 2, 3, 4, 5])
        for i, (key, values) in enumerate(all_attris.items()):
            axes[i + 1].hist(values, bins=bins, edgecolor='black')
            axes[i + 1].set_title(all_attris_names[i])
            if i == 5:
                axes[i + 1].set_xticks([0, 1])
            else:
                axes[i + 1].set_xticks([1, 2, 3, 4, 5])
        for j in range(len(all_attris) + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()

    elif args.dataset == "LIDC":
        train_loader, val_loader, test_loader = load_lidc(batch_size=args.batch_size,
                                                          resize_shape=args.resize_shape,
                                                          threeD=args.threeD,
                                                          splitnumber=args.split_number,
                                                          args=args)
        numattributes = next(iter(train_loader))[2].shape[1]
        print("#attributes=#caps : " + str(numattributes))
    elif args.dataset == "Chexbert":
        train_loader, val_loader, test_loader = load_chexbert(batch_size=args.batch_size,
                                                              resize_shape=args.resize_shape,
                                                              splitnumber=args.split_number,
                                                              args=args)
        numattributes = next(iter(train_loader))[2].shape[1]
        print("#attributes=#caps : " + str(numattributes))
    elif args.dataset == "derm7pt":
        train_loader, val_loader, test_loader = load_derm7pt(batch_size=args.batch_size,
                                                          resize_shape=args.resize_shape,
                                                          splitnumber=args.split_number,
                                                          args=args)
        numattributes = next(iter(train_loader))[2].shape[1]
        print("#attributes=#caps : " + str(numattributes))

    if args.test:
        if args.model_path == None:
            raise TypeError(
                "Please specify the path to model by setting the parameter --model_path=\"path_to_model\"")
        if args.epoch == None:
            raise TypeError("Please specify the epoch of chosen model by setting the parameter --epoch=[int]")
        path = args.model_path
        model = torch.load(path, weights_only=False)
        if not args.base_model in ["ViT","ResNetMT","DenseNetMT"]:
            model.args.train = False
            model.args.test = True

        print("run test split")
        _, test_acc, te_attr_acc = test(testmodel=model, data_loader=test_loader, args=args)
        print('test acc = ')
        print(test_acc)
        print("attr test acc = " + str(te_attr_acc))


        if args.dataset == "FunnyNodules":
            from test import test_contrastivity, visualize_all_attribute_responses, visualize_roundness_cs_interaction
            print("do test_contrastivity")
            test_contrastivity(model, args)
            print("do visualize_all_attribute_responses")
            visualize_all_attribute_responses(model, args)
            print("do visualize_roundness_cs_interaction")
            visualize_roundness_cs_interaction(model, args, args.resize_shape[0])

        if args.dataset == "LIDC":
            test_acc, test_attracc, test_dc = test_indepth(testmodel=model,
                                                           data_loader=test_loader,
                                                           epoch=args.epoch,
                                                           prototypefoldername=path.split("_")[0],
                                                           args=args)
            print("dc without exchange:" + str(test_dc))
            print("PE_test_acc (with use of prototypes) target_accuracy")
            print(test_acc)
            print("PE_test_attr_acc: " + str(test_attracc))
        elif args.dataset == "FunnyNodules":
            test_acc, test_attracc = test_indepth(testmodel=model,
                                                           data_loader=test_loader,
                                                           epoch=args.epoch,
                                                           prototypefoldername=path.split("_")[0],
                                                           args=args)
            print("PE_test_acc (with use of prototypes) target_accuracy")
            print(test_acc)
            print("PE_test_attr_acc: " + str(test_attracc))
        elif args.dataset == "Chexbert":
            test_acc, test_attracc = test_indepth(testmodel=model,
                                                  data_loader=test_loader,
                                                  epoch=args.epoch,
                                                  prototypefoldername=path.split("_")[0],
                                                  args=args)

            print("PE_test_acc (with use of prototypes) target_auc, target_accuracy, target_precision, target_recall, target_f1: ")
            print(test_acc)
            print("PE_test_attr_acc: " + str(test_attracc))




        if args.dataset == "LIDC":
            print("start test time intervention")
            from test_time_intervention import test_time_intervention
            test_time_intervention(testmodel=model, val_data_loader=val_loader, test_data_loader=test_loader,
                                   epoch=args.epoch, prototypefoldername=path.split("_")[0], args=args)

        if args.dataset in ["LIDC", "derm7pt", "FunnyNodules"]:
            test_show_inference(testmodel=model, data_loader=test_loader, epoch=args.epoch,
                                       prototypefoldername=path.split("_")[0], args=args)
        if args.dataset == "LIDC":
            test_indepth_attripredCorr(testmodel=model,
                                       train_loader=train_loader,
                                       test_loader=test_loader,
                                       epoch=args.epoch,
                                       prototypefoldername=path.split("_")[0],
                                       args=args)
            if args.base_model == "ConvNet":
                test_getweightedsharedinfo(testmodel=model,
                                           test_loader=test_loader,
                                           epoch=args.epoch,
                                           prototypefoldername=path.split("_")[0],
                                           args=args)

    if args.train:
        if args.model_path == None:
            epoch_start = 0
            if args.base_model=="ResNetMT":
                import torch
                import torchvision.models as models
                model = models.resnet50(pretrained=True)
                if args.dataset == "FunnyNodules":
                    num_classes = 7
                    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
                else: sys.exit()
            elif args.base_model=="DenseNetMT":
                import torch
                import torchvision.models as models
                model = models.densenet121(pretrained=True)
                if args.dataset == "FunnyNodules":
                    num_classes = 7
                    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
                else: sys.exit()
            elif args.base_model=="ViT":
                if args.dataset == "LIDC":
                    model = CustomVisionTransformer(args=args)
                elif args.dataset == "derm7pt":
                    model = CustomVisionTransformer(args=args)
                elif args.dataset == "FunnyNodules":
                    model = CustomVisionTransformer(args=args)
            else:
                model = ProtoCapsNet(input_size=[1, *args.resize_shape], numcaps=numattributes, routings=3,
                                     out_dim_caps=args.out_dim_caps, activation_fn="sigmoid", threeD=args.threeD,
                                     numProtos=args.num_protos, args=args)
        else:
            print(args.model_path)
            model = torch.load(args.model_path)
            print(args.model_path)
            epoch_start = int(args.model_path.split(".")[-2].split("_")[-1])+1

        import torch
        print(torch.__version__)  # sollte 2.5.1 oder ähnlich sein
        print(torch.version.cuda)  # sollte "12.1" anzeigen
        print(torch.cuda.is_available())  # sollte True sein
        print(torch.cuda.get_device_name(0))  # sollte "NVIDIA GeForce RTX 3090" sein
        model.cuda()
        print(model)
        if args.dataset == "derm7pt" and args.base_model == "ViT":
            opt_specs = [{'params': model.vit.parameters(), 'lr': args.lr},
                         {'params': model.encoder_attri0.parameters(), 'lr': args.lr},
                         {'params': model.encoder_attri1.parameters(), 'lr': args.lr},
                         {'params': model.encoder_attri2.parameters(), 'lr': args.lr},
                         {'params': model.encoder_attri3.parameters(), 'lr': args.lr},
                         {'params': model.encoder_attri4.parameters(), 'lr': args.lr},
                         {'params': model.encoder_attri5.parameters(), 'lr': args.lr},
                         {'params': model.head_attri0.parameters(), 'lr': args.lr},
                         {'params': model.head_attri1.parameters(), 'lr': args.lr},
                         {'params': model.head_attri2.parameters(), 'lr': args.lr},
                         {'params': model.head_attri3.parameters(), 'lr': args.lr},
                         {'params': model.head_attri4.parameters(), 'lr': args.lr},
                         {'params': model.head_attri5.parameters(), 'lr': args.lr},
                         {'params': model.encoder_tar.parameters(), 'lr': args.lr},
                         {'params': model.head_tar.parameters(), 'lr': args.lr},
                         {'params': model.fc8to1.parameters(), 'lr': args.lr},
                         {'params': model.decoderlayers.parameters(), 'lr': args.lr},
                         {'params': model.decoder_decon.parameters(), 'lr': args.lr},
                         {'params': model.decoder_norm.parameters(), 'lr': args.lr},
                         {'params': model.decoder_sigm.parameters(), 'lr': args.lr},
                         {'params': model.protodigis0, 'lr': 0.01},
                         {'params': model.protodigis1, 'lr': 0.01},
                         {'params': model.protodigis2, 'lr': 0.01},
                         {'params': model.protodigis3, 'lr': 0.01},
                         {'params': model.protodigis4, 'lr': 0.01},
                         {'params': model.protodigis5, 'lr': 0.01}]
            optimizer = torch.optim.Adam(opt_specs)
        else:
            print("LR= "+str(args.lr))
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        print("training samples: " + str(len(train_loader.dataset)))
        print("val samples: " + str(len(val_loader.dataset)))
        print("test samples: " + str(len(test_loader.dataset)))
        train_samples_with_attrLabels_Loss = torch.randperm(len(train_loader.dataset))[
                                             :int(args.shareAttrLabels * len(train_loader.dataset))]
        print(str(len(
            train_samples_with_attrLabels_Loss)) + " samples are being considered of having attribute labels")

        # During training model are being saved with the format
        # YYYY/MM/DD TIME_accuracyRAW_epochnumber.pth  -> best validation accuracy
        # YYYY/MM/DD TIME_accuracy_epochnumber.pth  -> best validation accuracy using prototypes during inference and respective prototypes in folder /prototypes

        wandb.init(project='FunnyNodules')
        wandb.watch(model)
        datestr = str(datetime.datetime.now())
        print("this run has datestr " + datestr)
        if not os.path.exists('prototypes'):
            os.makedirs('prototypes')
        protosavedir = "./prototypes/" + str(datestr)
        os.mkdir(protosavedir)
        overall_acc_best = 0.0
        best_val_acc = 0.0
        raw_overall_acc_best = 0.0
        earlyStopping_counter = 1
        earlyStopping_max = 100  # push iterations
        import time
        for ep in range(epoch_start,args.epochs):
            print("epoch number "+str(ep) +" "+str(time.time()))
            if ep % args.push_step == 0:
                if ep >= args.warmup:
                    print("Pushing")
                    print("epoch number "+str(ep) +" push start: "+str(time.time()))
                    model, mindists_X, mindists_alllabels, protos_total_id = pushprotos(model_push=model, data_loader=train_loader,
                                                                       idx_with_attri=train_samples_with_attrLabels_Loss,
                                                                       args=args)
                    print("epoch number "+str(ep) +" push end: "+str(time.time()))
                    protosavedir = "./prototypes/" + str(datestr) + "/" + str(ep)
                    os.mkdir(protosavedir)
                    for cpsi in range(len(mindists_X)):
                        for proto_idx in range(mindists_X[cpsi].shape[0]):
                            for proto_idx2 in range(mindists_X[cpsi].shape[1]):
                                if args.dataset == "derm7pt":
                                    np.save(os.path.join(protosavedir + "/",
                                                     "cpslnr" + str(cpsi) + "_protonr" + str(
                                                         proto_idx) + "-" + str(proto_idx2) + "_gtattrcs" + str(
                                                         mindists_alllabels[cpsi][proto_idx, proto_idx2])),
                                        mindists_X[cpsi][proto_idx, proto_idx2].permute(1,2,0))
                                else:
                                    np.save(os.path.join(protosavedir + "/",
                                                     "cpslnr" + str(cpsi) + "_protonr" + str(
                                                         proto_idx) + "-" + str(proto_idx2) + "_gtattrcs" + str(
                                                         mindists_alllabels[cpsi][proto_idx, proto_idx2])),
                                        mindists_X[cpsi][proto_idx, proto_idx2, 0])
                                if args.dataset == "LIDC" and protos_total_id[cpsi][proto_idx, proto_idx2, 0] >-1:
                                    total_image = Image.open("data_total_split"+str(args.split_number)+"/"+str(int(protos_total_id[cpsi][proto_idx, proto_idx2, 0].item()))+"_"+
                                                             str(int(protos_total_id[cpsi][proto_idx, proto_idx2, 1].item()))+"_"+
                                                             str(int(protos_total_id[cpsi][proto_idx, proto_idx2, 2].item()))+"_"+
                                                             str(int(protos_total_id[cpsi][proto_idx, proto_idx2, 3].item()))+".png")
                                    total_image.save(os.path.join(protosavedir + "/",
                                                         "total_"+"cpslnr" + str(cpsi) + "_protonr" + str(
                                                             proto_idx) + "-" + str(proto_idx2) + "_gtattrcs" + str(
                                                             mindists_alllabels[cpsi][proto_idx, proto_idx2])+".png"))
                    if args.dataset == "LIDC":
                        valwProtoE_acc, valwProtoE_attracc, _ = test_indepth(testmodel=model,
                                                                             data_loader=val_loader,
                                                                             epoch=ep,
                                                                             prototypefoldername=datestr,
                                                                             args=args)
                    elif args.dataset in ["Chexbert","derm7pt", "FunnyNodules"]:
                        valwProtoE_acc, valwProtoE_attracc = test_indepth(testmodel=model,
                                                                          data_loader=val_loader,
                                                                          epoch=ep,
                                                                          prototypefoldername=datestr,
                                                                          args=args)
                    if args.dataset == "LIDC":
                        wandb.log({'PE_val_acc': valwProtoE_acc, 'epoch': ep,
                                   'PE_val_acc_a0': valwProtoE_attracc[0], 'PE_val_acc_a1': valwProtoE_attracc[1],
                                   'PE_val_acc_a2': valwProtoE_attracc[2], 'PE_val_acc_a3': valwProtoE_attracc[3],
                                   'PE_val_acc_a4': valwProtoE_attracc[4], 'PE_val_acc_a5': valwProtoE_attracc[5],
                                   'PE_val_acc_a6': valwProtoE_attracc[6], 'PE_val_acc_a7': valwProtoE_attracc[7]})
                    elif args.dataset == "FunnyNodules":
                        wandb.log({'PE_val_acc': valwProtoE_acc, 'epoch': ep,
                                   'PE_val_acc_a0': valwProtoE_attracc[0], 'PE_val_acc_a1': valwProtoE_attracc[1],
                                   'PE_val_acc_a2': valwProtoE_attracc[2], 'PE_val_acc_a3': valwProtoE_attracc[3],
                                   'PE_val_acc_a4': valwProtoE_attracc[4], 'PE_val_acc_a5': valwProtoE_attracc[5]})
                    elif args.dataset == "derm7pt":
                        wandb.log({'PE_val_acc': valwProtoE_acc, 'epoch': ep,
                                   'PE_val_acc_a0': valwProtoE_attracc[0], 'PE_val_acc_a1': valwProtoE_attracc[1],
                                   'PE_val_acc_a2': valwProtoE_attracc[2], 'PE_val_acc_a3': valwProtoE_attracc[3],
                                   'PE_val_acc_a4': valwProtoE_attracc[4], 'PE_val_acc_a5': valwProtoE_attracc[5],
                                   'PE_val_acc_a6': valwProtoE_attracc[6]})
                    elif args.dataset == "Chexbert":
                        wandb.log({'PE_val_acc': valwProtoE_acc[0], 'epoch': ep,
                                   'PE_val_acc_a0': valwProtoE_attracc[0], 'PE_val_acc_a1': valwProtoE_attracc[1],
                                   'PE_val_acc_a2': valwProtoE_attracc[2], 'PE_val_acc_a3': valwProtoE_attracc[3],
                                   'PE_val_acc_a4': valwProtoE_attracc[4], 'PE_val_acc_a5': valwProtoE_attracc[5],
                                   'PE_val_acc_a6': valwProtoE_attracc[6], 'PE_val_acc_a7': valwProtoE_attracc[7],
                                   'PE_val_acc_a8': valwProtoE_attracc[8], 'PE_val_acc_a9': valwProtoE_attracc[9],
                                   'PE_val_acc_a10': valwProtoE_attracc[10],
                                   'PE_val_acc_a11': valwProtoE_attracc[11],
                                   'PE_val_acc_a12': valwProtoE_attracc[12]})

                    print("PE_val_acc: ")
                    print(valwProtoE_acc)
                    print("PE_val_attracc: " + str(valwProtoE_attracc))

                    if args.dataset == "LIDC":
                        overall_acc = 8 * valwProtoE_acc + sum(valwProtoE_attracc)
                    elif args.dataset == "derm7pt":
                        overall_acc = 7 * valwProtoE_acc + sum(valwProtoE_attracc)
                    elif args.dataset == "FunnyNodules":
                        overall_acc = 6 * valwProtoE_acc + sum(valwProtoE_attracc)
                    elif args.dataset == "Chexbert":
                        overall_acc = 13 * valwProtoE_acc[0] + sum(valwProtoE_attracc)
                    if overall_acc > overall_acc_best:
                        if args.dataset in ["LIDC","derm7pt","FunnyNodules"]:
                            torch.save(model, str(datestr) + "_" + str(valwProtoE_acc) + "_" + str(ep) + '.pth')
                            print("Save new best model with path: " + str(datestr) + "_" + str(
                                valwProtoE_acc) + "_" + str(
                                ep) + '.pth')
                        elif args.dataset == "Chexbert":
                            torch.save(model, str(datestr) + "_" + str(valwProtoE_acc[0]) + "_" + str(ep) + '.pth')
                            print("Save new best model with path: " + str(datestr) + "_" + str(
                                valwProtoE_acc[0]) + "_" + str(
                                ep) + '.pth')
                        overall_acc_best = overall_acc
                    if overall_acc_best > best_val_acc:
                        best_val_acc = overall_acc_best
                        earlyStopping_counter = 1
                    else:
                        earlyStopping_counter += 1
                        if earlyStopping_counter > earlyStopping_max:
                            sys.exit()
            print("Training")
            print("epoch number "+str(ep) +" training start: "+str(time.time()))
            model, tr_acc, tr_attr_acc = train_model(
                model, train_loader, args, epoch=ep, optim=optimizer,
                idx_with_attri=train_samples_with_attrLabels_Loss)
            print("epoch number "+str(ep) +" training end: "+str(time.time()))
            print("train acc = ")
            print(tr_acc)
            if args.dataset == "LIDC":
                wandb.log({'train_acc': tr_acc, 'epoch': ep,
                           'train_acc_a0': tr_attr_acc[0], 'train_acc_a1': tr_attr_acc[1],
                           'train_acc_a2': tr_attr_acc[2],
                           'train_acc_a3': tr_attr_acc[3], 'train_acc_a4': tr_attr_acc[4],
                           'train_acc_a5': tr_attr_acc[5],
                           'train_acc_a6': tr_attr_acc[6], 'train_acc_a7': tr_attr_acc[7]})
            elif args.dataset == "FunnyNodules":
                wandb.log({'train_acc': tr_acc, 'epoch': ep,
                           'train_acc_a0': tr_attr_acc[0], 'train_acc_a1': tr_attr_acc[1],
                           'train_acc_a2': tr_attr_acc[2],
                           'train_acc_a3': tr_attr_acc[3], 'train_acc_a4': tr_attr_acc[4],
                           'train_acc_a5': tr_attr_acc[5]})
            elif args.dataset == "derm7pt":
                wandb.log({'train_acc': tr_acc, 'epoch': ep,
                           'train_acc_a0': tr_attr_acc[0], 'train_acc_a1': tr_attr_acc[1],
                           'train_acc_a2': tr_attr_acc[2],
                           'train_acc_a3': tr_attr_acc[3], 'train_acc_a4': tr_attr_acc[4],
                           'train_acc_a5': tr_attr_acc[5],'train_acc_a6': tr_attr_acc[6]})
            elif args.dataset == "Chexbert":
                wandb.log({'train_acc': tr_acc[0], 'epoch': ep,
                           'train_acc_a0': tr_attr_acc[0], 'train_acc_a1': tr_attr_acc[1],
                           'train_acc_a2': tr_attr_acc[2], 'train_acc_a3': tr_attr_acc[3],
                           'train_acc_a4': tr_attr_acc[4], 'train_acc_a5': tr_attr_acc[5],
                           'train_acc_a6': tr_attr_acc[6], 'train_acc_a7': tr_attr_acc[7],
                           'train_acc_a8': tr_attr_acc[8], 'train_acc_a9': tr_attr_acc[9],
                           'train_acc_a10': tr_attr_acc[10], 'train_acc_a11': tr_attr_acc[11],
                           'train_acc_a12': tr_attr_acc[12], })
            print("Validation")
            val_loss, val_acc, val_attr_acc = test(testmodel=model, data_loader=val_loader, args=args)
            print('val acc = ')
            print(val_acc)
            print("val_attr_acc = " + str(val_attr_acc))
            print('val loss = ')
            print(val_loss)
            wandb.log({'val_loss': val_loss})
            if args.base_model=="ViT":
                if args.dataset == "LIDC":
                    if args.onlyTar:
                        if val_acc > raw_overall_acc_best:
                            raw_overall_acc_best = val_acc
                            torch.save(model, str(datestr) + "RAW_" + str(val_acc) + "_" + str(ep) + '.pth')
                            print("RAW Save new best model with path: " + str(datestr) + "_" + str(
                                val_acc) + "_" + str(ep) + '.pth')
                    else:
                        if (8 * val_acc + sum(val_attr_acc))> raw_overall_acc_best:
                            raw_overall_acc_best = (8 * val_acc + sum(val_attr_acc))
                            torch.save(model, str(datestr) + "RAW_" + str(val_acc) + "_" + str(ep) + '.pth')
                            print("RAW Save new best model with path: " + str(datestr) + "_" + str(val_acc) + "_" + str(ep) + '.pth')
                elif args.dataset == "derm7pt":
                    if (7 * val_acc + sum(val_attr_acc))> raw_overall_acc_best:
                        raw_overall_acc_best = (7 * val_acc + sum(val_attr_acc))
                        torch.save(model, str(datestr) + "RAW_" + str(val_acc) + "_" + str(ep) + '.pth')
                        print("RAW Save new best model with path: " + str(datestr) + "_" + str(val_acc) + "_" + str(ep) + '.pth')
                elif args.dataset == "FunnyNodules":
                    if (6 * val_acc + sum(val_attr_acc)) > raw_overall_acc_best:
                        raw_overall_acc_best = (6 * val_acc + sum(val_attr_acc))
                        torch.save(model, str(datestr) + "RAW_" + str(val_acc) + "_" + str(ep) + '.pth')
                        print("RAW Save new best model with path: " + str(datestr) + "_" + str(val_acc) + "_" + str(
                            ep) + '.pth')
            else:
                if args.dataset == "derm7pt":
                    if (7 * val_acc + sum(val_attr_acc))> raw_overall_acc_best:
                        raw_overall_acc_best = (7 * val_acc + sum(val_attr_acc))
                        torch.save(model, str(datestr) + "RAW_" + str(val_acc) + "_" + str(ep) + '.pth')
                        print("RAW Save new best model with path: " + str(datestr) + "_" + str(val_acc) + "_" + str(ep) + '.pth')
                elif args.dataset == "FunnyNodules":
                    if (6 * val_acc + sum(val_attr_acc)) > raw_overall_acc_best:
                        raw_overall_acc_best = (6 * val_acc + sum(val_attr_acc))
                        torch.save(model, str(datestr) + "RAW_" + str(val_acc) + "_" + str(ep) + '.pth')
                        print("RAW Save new best model with path: " + str(datestr) + "_" + str(val_acc) + "_" + str(
                            ep) + '.pth')
            if args.dataset == "LIDC":
                wandb.log({'val_acc': val_acc, 'epoch': ep,
                           'val_acc_a0': val_attr_acc[0], 'val_acc_a1': val_attr_acc[1],
                           'val_acc_a2': val_attr_acc[2],
                           'val_acc_a3': val_attr_acc[3], 'val_acc_a4': val_attr_acc[4],
                           'val_acc_a5': val_attr_acc[5],
                           'val_acc_a6': val_attr_acc[6], 'val_acc_a7': val_attr_acc[7]})
            elif args.dataset == "FunnyNodules":
                wandb.log({'val_acc': val_acc, 'epoch': ep,
                           'val_acc_a0': val_attr_acc[0], 'val_acc_a1': val_attr_acc[1],
                           'val_acc_a2': val_attr_acc[2],
                           'val_acc_a3': val_attr_acc[3], 'val_acc_a4': val_attr_acc[4],
                           'val_acc_a5': val_attr_acc[5]})
            elif args.dataset == "derm7pt":
                wandb.log({'val_acc': val_acc, 'epoch': ep,
                           'val_acc_a0': val_attr_acc[0], 'val_acc_a1': val_attr_acc[1],
                           'val_acc_a2': val_attr_acc[2],
                           'val_acc_a3': val_attr_acc[3], 'val_acc_a4': val_attr_acc[4],
                           'val_acc_a5': val_attr_acc[5],'val_acc_a6': val_attr_acc[6]})
            elif args.dataset == "Chexbert":
                wandb.log({'val_acc': val_acc[0], 'epoch': ep,
                           'val_acc_a0': val_attr_acc[0], 'val_acc_a1': val_attr_acc[1],
                           'val_acc_a2': val_attr_acc[2],
                           'val_acc_a3': val_attr_acc[3], 'val_acc_a4': val_attr_acc[4],
                           'val_acc_a5': val_attr_acc[5],
                           'val_acc_a6': val_attr_acc[6], 'val_acc_a7': val_attr_acc[7],
                           'val_acc_a8': val_attr_acc[8],
                           'val_acc_a9': val_attr_acc[9], 'val_acc_a10': val_attr_acc[10],
                           'val_acc_a11': val_attr_acc[11],
                           'val_acc_a12': val_attr_acc[12]})
            print("Epoch " + str(ep) + ' ' + '-' * 70)
