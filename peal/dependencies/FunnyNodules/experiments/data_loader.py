"""
Loading and preprocessing of datasets
    - LIDC-IDRI
    - derm7pt
    - ChexBert

Author: Luisa Gallée, Github: `https://github.com/XRad-Ulm/HierViT`
"""

import sys
import numpy
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
import pylidc as pl
import os.path
import h5py
import pandas as pd

def generateNoduleSplitFiles():
    """
    Generate data splits nodule and scan wise stratified regarding malignancy.
    """
    scan = pl.query(pl.Scan).all()
    nod_list = []
    nod_meanmal_list = []
    scan_i_list = []
    totalnod = 0
    for scani in range(len(scan)):
        nodules = scan[scani].cluster_annotations()
        print("Generate split files: The scan " + str(scani) + " has %d nodules." % len(nodules))
        for i, nod in enumerate(nodules):
            if len(nod) >= 3 and len(nod) <= 4:
                nod_mal = []
                for nod_anni in range(len(nod)):
                    nod_mal.append(nod[nod_anni].malignancy)
                if np.mean(nod_mal) != 3.0:
                    nod_list.append(totalnod)
                    totalnod += 1
                    nod_meanmal_list.append(np.mean(nod_mal))
                    scan_i_list.append(scani)

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=12)
    foldsplit = skf.split(nod_list, np.rint(nod_meanmal_list), groups=scan_i_list)

    for i, (train_index, test_index) in enumerate(foldsplit):
        np.save('LIDC_split' + str(i) + '_train', train_index)
        np.save('LIDC_split' + str(i) + '_test', test_index)


def generateDatasetforSplit(splitnumber, threeD, resize_shape):
    """
    Generate and save data set for specified split. Preprocessing to image, mask and labels of nodule.
    :param splitnumber: number of split to be processed
    :param threeD: Bool, True: for cropping out volume, False: for cropping out center slice
    :param resize_shape: resize shape of slice/volume
    """
    split_train = np.load("LIDC_split" + str(splitnumber) + "_train.npy")
    split_test = np.load("LIDC_split" + str(splitnumber) + "_test.npy")
    scan = pl.query(pl.Scan).all()
    img_list_total_train, img_list_total_test = [], []
    img_list_train, img_list_test = [], []
    mask_list_train, mask_list_test = [], []
    label_list_train, label_list_test = [], []
    max_x = 0
    max_y = 0
    max_z = 0
    totalnod = 0
    if not os.path.exists(f"data_total_split{splitnumber}"):
        os.makedirs(f"data_total_split{splitnumber}")
    for scani in range(len(scan)):
        print(str(scani)+"/"+str(len(scan)))
        nodules = scan[scani].cluster_annotations()
        print("Generate dataset for split "+str(splitnumber)+": The scan " + str(scani) + " has %d nodules." % len(nodules))
        for i, nod in enumerate(nodules):
            if len(nod) >= 3 and len(nod) <= 4:
                nod_sub, nod_int, nod_calc, nod_spher, nod_mar, nod_lob, nod_spic, nod_tex = [], [], [], [], [], [], [], []
                nod_mal = []
                for nod_anni in range(len(nod)):
                    nod_sub.append(nod[nod_anni].subtlety)
                    nod_int.append(nod[nod_anni].internalStructure)
                    nod_calc.append(nod[nod_anni].calcification)
                    nod_spher.append(nod[nod_anni].sphericity)
                    nod_mar.append(nod[nod_anni].margin)
                    nod_lob.append(nod[nod_anni].lobulation)
                    nod_spic.append(nod[nod_anni].spiculation)
                    nod_tex.append(nod[nod_anni].texture)
                    nod_mal.append(nod[nod_anni].malignancy)
                if np.mean(nod_mal) != 3.0:
                    for nod_anni in range(len(nod)):
                        labels = [nod[nod_anni].subtlety, nod[nod_anni].internalStructure,
                                  nod[nod_anni].calcification,
                                  nod[nod_anni].sphericity, nod[nod_anni].margin, nod[nod_anni].lobulation,
                                  nod[nod_anni].spiculation, nod[nod_anni].texture, nod[nod_anni].malignancy,
                                  np.mean(nod_sub), np.std(nod_sub), np.mean(nod_int), np.std(nod_int),
                                  np.mean(nod_calc), np.std(nod_calc),
                                  np.mean(nod_spher), np.std(nod_spher), np.mean(nod_mar), np.std(nod_mar),
                                  np.mean(nod_lob), np.std(nod_lob),
                                  np.mean(nod_spic), np.std(nod_spic), np.mean(nod_tex), np.std(nod_tex),
                                  np.mean(nod_mal), np.std(nod_mal)]

                        mask = nod[nod_anni].boolean_mask()
                        bbox = nod[nod_anni].bbox()
                        vol = nod[nod_anni].scan.to_volume()

                        #APPLY LUNG WINDOW
                        window_level = -600
                        window_width = 1600
                        window_ub = window_level + (window_width/2)
                        window_lb = window_level - (window_width/2)
                        vol[vol > window_ub] = window_ub
                        vol[vol < window_lb] = window_lb

                        centroid = nod[nod_anni].centroid.astype(int)
                        entire_mask = np.zeros_like(vol)
                        entire_mask[bbox] = mask

                        cropout_hw = max([mask.shape[0], mask.shape[1]])
                        cropout_d = resize_shape[-1]
                        cropout_size = [cropout_hw, cropout_hw, cropout_d]
                        cropout_size_half = [x//2 for x in cropout_size]
                        cropout_border = np.array(
                            [[0, vol.shape[0]], [0, vol.shape[1]], [0, vol.shape[2]]])
                        for d in range(3):
                            if int(centroid[d] - cropout_size_half[d]) < 0 or int(
                                    centroid[d] + cropout_size_half[d]) > vol.shape[d]:
                                if int(centroid[d] - cropout_size_half[d]) < 0:
                                    cropout_border[d, 1] = cropout_size[d]
                                else:
                                    cropout_border[d, 0] = vol.shape[d] - cropout_size[d]
                            else:
                                cropout_border[d, 0] = int(centroid[d] - cropout_size_half[d])
                                cropout_border[d, 1] = int(centroid[d] + cropout_size_half[d])
                        if not threeD:
                            new_img_total = vol[:,:,bbox[2].start:bbox[2].stop]
                            new_img = vol[cropout_border[0, 0]:cropout_border[0, 1],
                                      cropout_border[1, 0]:cropout_border[1, 1],
                                      bbox[2].start:bbox[2].stop]
                            new_mask = entire_mask[cropout_border[0, 0]:cropout_border[0, 1],
                                       cropout_border[1, 0]:cropout_border[1, 1],
                                       bbox[2].start:bbox[2].stop]
                        else:
                            new_img = vol[cropout_border[0, 0]:cropout_border[0, 1],
                                      cropout_border[1, 0]:cropout_border[1, 1],
                                      cropout_border[2, 0]:cropout_border[2, 1]]
                            new_mask = entire_mask[cropout_border[0, 0]:cropout_border[0, 1],
                                       cropout_border[1, 0]:cropout_border[1, 1],
                                       cropout_border[2, 0]:cropout_border[2, 1]]
                        if not threeD:
                            for slice_i in range(new_mask.shape[-1]):
                                img_PIL = Image.fromarray(new_img[:, :, slice_i])
                                mask_PIL = Image.fromarray(new_mask[:, :, slice_i])
                                out_img = img_PIL.resize(resize_shape)
                                out_mask = mask_PIL.resize(resize_shape)
                                img_nparray = np.asarray(out_img, dtype=np.int16)
                                mask_nparray = np.asarray(out_mask, dtype=np.uint8)
                                from matplotlib import pyplot as plt
                                plt.figure()
                                plt.imshow(new_img_total[:, :, slice_i], cmap="gray")
                                xs = [cropout_border[1, 0], cropout_border[1, 1], cropout_border[1, 1], cropout_border[1, 0], cropout_border[1, 0]]
                                ys = [cropout_border[0, 0], cropout_border[0, 0], cropout_border[0, 1], cropout_border[0, 1], cropout_border[0, 0]]
                                plt.plot(xs, ys, color="red", linewidth=0.5)
                                plt.axis('off')
                                plt.savefig("data_total_split"+str(splitnumber)+"/"+str(scani)+"_"+str(i)+"_"+str(nod_anni)+"_"+str(slice_i), bbox_inches='tight')
                                plt.close()
                                if totalnod in split_train:
                                    img_list_train.append(img_nparray)
                                    mask_list_train.append(mask_nparray)
                                    label_list_train.append(labels)
                                    img_list_total_train.append(str(scani)+"_"+str(i)+"_"+str(nod_anni)+"_"+str(slice_i))
                                elif totalnod in split_test:
                                    img_list_test.append(img_nparray)
                                    mask_list_test.append(mask_nparray)
                                    label_list_test.append(labels)
                                    img_list_total_test.append(str(scani)+"_"+str(i)+"_"+str(nod_anni)+"_"+str(slice_i))
                        else:
                            new_img_resized = numpy.zeros((resize_shape[2],resize_shape[0],resize_shape[1]))
                            new_mask_resized = numpy.zeros((resize_shape[2],resize_shape[0],resize_shape[1]))
                            for slice_i in range(new_mask_resized.shape[0]):
                                img_PIL = Image.fromarray(new_img[:, :, slice_i])
                                mask_PIL = Image.fromarray(new_mask[:, :, slice_i])
                                out_img = img_PIL.resize([resize_shape[0],resize_shape[1]])
                                out_mask = mask_PIL.resize([resize_shape[0],resize_shape[1]])
                                new_img_resized[slice_i] = np.asarray(out_img, dtype=np.int16)
                                new_mask_resized[slice_i] = np.asarray(out_mask, dtype=np.uint8)

                            if totalnod in split_train:
                                img_list_train.append(new_img_resized)
                                mask_list_train.append(new_mask_resized)
                                label_list_train.append(labels)
                            elif totalnod in split_test:
                                img_list_test.append(new_img_resized)
                                mask_list_test.append(new_mask_resized)
                                label_list_test.append(labels)

                            if (totalnod > 0) and (totalnod % 20 == 0):
                                if len(img_list_train) > 0:
                                    img_train, mask_train, label_train = np.asarray(img_list_train), np.asarray(
                                        mask_list_train), np.asarray(
                                        label_list_train)
                                    if not os.path.isfile("3d_train_split" + str(splitnumber) + ".h5"):
                                        with h5py.File("3d_train_split" + str(splitnumber) + ".h5", 'w') as h5f:
                                            h5f.create_dataset("img", shape=(
                                                0, img_train.shape[1], img_train.shape[2], img_train.shape[3]),
                                                               chunks=True,
                                                               maxshape=(None, img_train.shape[1], img_train.shape[2],
                                                                         img_train.shape[3]))
                                            h5f.create_dataset("mask", shape=(
                                                0, mask_train.shape[1], mask_train.shape[2], mask_train.shape[3]),
                                                               chunks=True,
                                                               maxshape=(None, mask_train.shape[1], mask_train.shape[2],
                                                                         mask_train.shape[3]))
                                            h5f.create_dataset("label", shape=(
                                                0, label_train.shape[1]), chunks=True,
                                                               maxshape=(None, label_train.shape[1]))
                                    with h5py.File("3d_train_split" + str(splitnumber) + ".h5", 'a') as h5f:
                                        h5f["img"].resize((h5f["img"].shape[0] + img_train.shape[0]), axis=0)
                                        h5f["img"][-img_train.shape[0]:] = img_train
                                        h5f["mask"].resize((h5f["mask"].shape[0] + mask_train.shape[0]), axis=0)
                                        h5f["mask"][-mask_train.shape[0]:] = mask_train
                                        h5f["label"].resize((h5f["label"].shape[0] + label_train.shape[0]), axis=0)
                                        h5f["label"][-label_train.shape[0]:] = label_train
                                if len(img_list_test) > 0:
                                    img_test, mask_test, label_test = np.asarray(img_list_test), np.asarray(
                                        mask_list_test), np.asarray(
                                        label_list_test)
                                    if not os.path.isfile("3d_test_split" + str(splitnumber) + ".h5"):
                                        with h5py.File("3d_test_split" + str(splitnumber) + ".h5", 'w') as h5f:
                                            h5f.create_dataset("img", shape=(
                                                0, img_test.shape[1], img_test.shape[2], img_test.shape[3]),
                                                               chunks=True,
                                                               maxshape=(None, img_test.shape[1], img_test.shape[2],
                                                                         img_test.shape[3]))
                                            h5f.create_dataset("mask", shape=(
                                                0, mask_test.shape[1], mask_test.shape[2], mask_test.shape[3]),
                                                               chunks=True,
                                                               maxshape=(None, mask_test.shape[1], mask_test.shape[2],
                                                                         mask_test.shape[3]))
                                            h5f.create_dataset("label", shape=(
                                                0, label_test.shape[1]), chunks=True,
                                                               maxshape=(None, label_test.shape[1]))
                                    with h5py.File("3d_test_split" + str(splitnumber) + ".h5", 'a') as h5f:
                                        h5f["img"].resize((h5f["img"].shape[0] + img_test.shape[0]), axis=0)
                                        h5f["img"][-img_test.shape[0]:] = img_test
                                        h5f["mask"].resize((h5f["mask"].shape[0] + mask_test.shape[0]), axis=0)
                                        h5f["mask"][-mask_test.shape[0]:] = mask_test
                                        h5f["label"].resize((h5f["label"].shape[0] + label_test.shape[0]), axis=0)
                                        h5f["label"][-label_test.shape[0]:] = label_test

                                img_list_train, img_list_test = [], []
                                mask_list_train, mask_list_test = [], []
                                label_list_train, label_list_test = [], []
                    totalnod += 1

    img_train, mask_train, label_train, img_total_train = np.asarray(img_list_train), np.asarray(mask_list_train), np.asarray(
        label_list_train), np.asarray(img_list_total_train)
    img_test, mask_test, label_test, img_total_test = np.asarray(img_list_test), np.asarray(mask_list_test), np.asarray(
        label_list_test), np.asarray(img_list_total_test)
    if not threeD:
        np.save("LIDC_img_train_split" + str(splitnumber), img_train)
        np.save("LIDC_mask_train_split" + str(splitnumber), mask_train)
        np.save("LIDC_label_train_split" + str(splitnumber), label_train)
        np.save("LIDC_img_total_train_split" + str(splitnumber), img_total_train)
        np.save("LIDC_img_test_split" + str(splitnumber), img_test)
        np.save("LIDC_mask_test_split" + str(splitnumber), mask_test)
        np.save("LIDC_label_test_split" + str(splitnumber), label_test)
        np.save("LIDC_img_total_test_split" + str(splitnumber), img_total_test)
    else:
        if len(img_list_train) > 0:
            img_train, mask_train, label_train = np.asarray(img_list_train), np.asarray(
                mask_list_train), np.asarray(
                label_list_train)
            with h5py.File("3d_train_split" + str(splitnumber) + ".h5", 'a') as h5f:
                h5f["img"].resize((h5f["img"].shape[0] + img_train.shape[0]), axis=0)
                h5f["img"][-img_train.shape[0]:] = img_train
                h5f["mask"].resize((h5f["mask"].shape[0] + mask_train.shape[0]), axis=0)
                h5f["mask"][-mask_train.shape[0]:] = mask_train
                h5f["label"].resize((h5f["label"].shape[0] + label_train.shape[0]), axis=0)
                h5f["label"][-label_train.shape[0]:] = label_train
        if len(img_list_test) > 0:
            img_test, mask_test, label_test = np.asarray(img_list_test), np.asarray(
                mask_list_test), np.asarray(
                label_list_test)
            with h5py.File("3d_test_split" + str(splitnumber) + ".h5", 'a') as h5f:
                h5f["img"].resize((h5f["img"].shape[0] + img_test.shape[0]), axis=0)
                h5f["img"][-img_test.shape[0]:] = img_test
                h5f["mask"].resize((h5f["mask"].shape[0] + mask_test.shape[0]), axis=0)
                h5f["mask"][-mask_test.shape[0]:] = mask_test
                h5f["label"].resize((h5f["label"].shape[0] + label_test.shape[0]), axis=0)
                h5f["label"][-label_test.shape[0]:] = label_test


class ChexbertDataset(Dataset):
    def __init__(self, img_predir, img_dirs, resize_shape, img_labels, img_attributes):
        self.img_predir = img_predir
        self.img_dirs = img_dirs
        self.resize_shape = resize_shape
        self.img_labels = img_labels
        self.img_attributes = img_attributes
        self.counter = np.arange(len(img_labels))
    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, idx):
        image = np.expand_dims(np.asarray(Image.open(self.img_predir + self.img_dirs[idx]).resize(self.resize_shape))/256, axis=0)
        return image, self.img_labels[idx], self.img_attributes[idx], self.counter[idx]
def load_chexbert(batch_size,
                  resize_shape,
                  splitnumber,
                  args):
    """
    Generates DataLoader from Chexbert.
    :param batch_size: batch size of returning DataLoader
    :param resize_shape: resize shape of twoD slice
    :return: Dataloaders of train, validation and test data
    """
    dataset_prepath="/path/to/directory containing CheXpert-v1.0/"

    train_file = dataset_prepath+"train_cheXbert.csv"


    df = pd.read_csv(train_file)
    train_frontallateral = df['Frontal/Lateral'].tolist()
    frontal_indexes = [i for i, x in enumerate(train_frontallateral) if x == 'Frontal']
    train_imagefile = np.array(df['Path'].tolist())[frontal_indexes]
    train_EnlargedCardiomediastinum = np.nan_to_num(np.array(df['Enlarged Cardiomediastinum'].tolist())[frontal_indexes])
    train_Cardiomegaly = np.nan_to_num(np.array(df['Cardiomegaly'].tolist())[frontal_indexes])
    train_LungOpacity = np.nan_to_num(np.array(df['Lung Opacity'].tolist())[frontal_indexes])
    train_LungLesion = np.nan_to_num(np.array(df['Lung Lesion'].tolist())[frontal_indexes])
    train_Edema = np.nan_to_num(np.array(df['Edema'].tolist())[frontal_indexes])
    train_Consolidation = np.nan_to_num(np.array(df['Consolidation'].tolist())[frontal_indexes])
    train_Pneumonia = np.nan_to_num(np.array(df['Pneumonia'].tolist())[frontal_indexes])
    train_Atelectasis = np.nan_to_num(np.array(df['Atelectasis'].tolist())[frontal_indexes])
    train_Pneumothorax = np.nan_to_num(np.array(df['Pneumothorax'].tolist())[frontal_indexes])
    train_PleuralEffusion = np.nan_to_num(np.array(df['Pleural Effusion'].tolist())[frontal_indexes])
    train_PleuralOther = np.nan_to_num(np.array(df['Pleural Other'].tolist())[frontal_indexes])
    train_Fracture = np.nan_to_num(np.array(df['Fracture'].tolist())[frontal_indexes])
    train_SupportDevices = np.nan_to_num(np.array(df['Support Devices'].tolist())[frontal_indexes])
    train_No_Finding = np.nan_to_num(np.array(df['No Finding'].tolist())[frontal_indexes])
    train_EnlargedCardiomediastinum[train_EnlargedCardiomediastinum == -1] = 1
    train_Cardiomegaly[train_Cardiomegaly == -1] = 1
    train_LungOpacity[train_LungOpacity == -1] = 1
    train_LungLesion[train_LungLesion == -1] = 1
    train_Edema[train_Edema == -1] = 1
    train_Consolidation[train_Consolidation == -1] = 1
    train_Pneumonia[train_Pneumonia == -1] = 1
    train_Atelectasis[train_Atelectasis == -1] = 1
    train_Pneumothorax[train_Pneumothorax == -1] = 1
    train_PleuralEffusion[train_PleuralEffusion == -1] = 1
    train_PleuralOther[train_PleuralOther == -1] = 1
    train_Fracture[train_Fracture == -1] = 1
    train_SupportDevices[train_SupportDevices == -1] = 1
    train_No_Finding[train_No_Finding == -1] = 1
    train_Finding_manual = (train_EnlargedCardiomediastinum+train_Cardiomegaly+train_LungOpacity+train_LungLesion+
                               train_Edema+train_Consolidation+train_Pneumonia+train_Atelectasis+train_Pneumothorax+
                               train_PleuralEffusion+train_PleuralOther+train_Fracture+train_SupportDevices)
    train_Finding_manual[train_Finding_manual>1]=1
    # Use train_No_finding_manual as target prediction
    train_labels = train_Finding_manual
    train_attri = np.stack([train_EnlargedCardiomediastinum,train_Cardiomegaly,train_LungOpacity,train_LungLesion,train_Edema,
                    train_Consolidation,train_Pneumonia,train_Atelectasis,train_Pneumothorax,train_PleuralEffusion,
                    train_PleuralOther,train_Fracture,train_SupportDevices], axis=-1)

    split_train, split_val = train_test_split(torch.arange(train_labels.shape[0]), test_size=0.1,
                                              stratify=train_labels)
    val_imagefile = train_imagefile[split_val]
    val_attri = train_attri[split_val]
    val_labels = train_labels[split_val]
    train_imagefile = train_imagefile[split_train]
    train_attri = train_attri[split_train]
    train_labels = train_labels[split_train]

    test_file = dataset_prepath+"CheXpert-v1.0/valid.csv"
    df = pd.read_csv(test_file)
    test_frontallateral = df['Frontal/Lateral'].tolist()
    frontal_indexes = [i for i, x in enumerate(test_frontallateral) if x == 'Frontal']
    test_imagefile = np.array(df['Path'].tolist())[frontal_indexes]
    test_EnlargedCardiomediastinum = np.nan_to_num(np.array(df['Enlarged Cardiomediastinum'].tolist())[frontal_indexes])
    test_Cardiomegaly = np.nan_to_num(np.array(df['Cardiomegaly'].tolist())[frontal_indexes])
    test_LungOpacity = np.nan_to_num(np.array(df['Lung Opacity'].tolist())[frontal_indexes])
    test_LungLesion = np.nan_to_num(np.array(df['Lung Lesion'].tolist())[frontal_indexes])
    test_Edema = np.nan_to_num(np.array(df['Edema'].tolist())[frontal_indexes])
    test_Consolidation = np.nan_to_num(np.array(df['Consolidation'].tolist())[frontal_indexes])
    test_Pneumonia = np.nan_to_num(np.array(df['Pneumonia'].tolist())[frontal_indexes])
    test_Atelectasis = np.nan_to_num(np.array(df['Atelectasis'].tolist())[frontal_indexes])
    test_Pneumothorax = np.nan_to_num(np.array(df['Pneumothorax'].tolist())[frontal_indexes])
    test_PleuralEffusion = np.nan_to_num(np.array(df['Pleural Effusion'].tolist())[frontal_indexes])
    test_PleuralOther = np.nan_to_num(np.array(df['Pleural Other'].tolist())[frontal_indexes])
    test_Fracture = np.nan_to_num(np.array(df['Fracture'].tolist())[frontal_indexes])
    test_SupportDevices = np.nan_to_num(np.array(df['Support Devices'].tolist())[frontal_indexes])
    test_No_Finding = np.nan_to_num(np.array(df['No Finding'].tolist())[frontal_indexes])
    test_EnlargedCardiomediastinum[test_EnlargedCardiomediastinum == -1] = 1
    test_Cardiomegaly[test_Cardiomegaly == -1] = 1
    test_LungOpacity[test_LungOpacity == -1] = 1
    test_LungLesion[test_LungLesion == -1] = 1
    test_Edema[test_Edema == -1] = 1
    test_Consolidation[test_Consolidation == -1] = 1
    test_Pneumonia[test_Pneumonia == -1] = 1
    test_Atelectasis[test_Atelectasis == -1] = 1
    test_Pneumothorax[test_Pneumothorax == -1] = 1
    test_PleuralEffusion[test_PleuralEffusion == -1] = 1
    test_PleuralOther[test_PleuralOther == -1] = 1
    test_Fracture[test_Fracture == -1] = 1
    test_SupportDevices[test_SupportDevices == -1] = 1
    test_No_Finding[test_No_Finding == -1] = 1
    test_Finding_manual = (
                test_EnlargedCardiomediastinum + test_Cardiomegaly + test_LungOpacity + test_LungLesion +
                test_Edema + test_Consolidation + test_Pneumonia + test_Atelectasis + test_Pneumothorax +
                test_PleuralEffusion + test_PleuralOther + test_Fracture + test_SupportDevices)
    test_Finding_manual[test_Finding_manual > 1] = 1
    test_labels = test_Finding_manual
    test_attri = np.stack([test_EnlargedCardiomediastinum,test_Cardiomegaly,test_LungOpacity,test_LungLesion,test_Edema,
                    test_Consolidation,test_Pneumonia,test_Atelectasis,test_Pneumothorax,test_PleuralEffusion,
                    test_PleuralOther,test_Fracture,test_SupportDevices], axis=-1)


    print(train_attri.shape)
    print(train_labels.shape)
    print(val_attri.shape)
    print(val_labels.shape)
    print(test_attri.shape)
    print(test_labels.shape)

    train_loader = torch.utils.data.DataLoader(ChexbertDataset(dataset_prepath, train_imagefile, resize_shape, train_labels, train_attri), batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(ChexbertDataset(dataset_prepath, val_imagefile, resize_shape, val_labels, val_attri), batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(ChexbertDataset(dataset_prepath, test_imagefile, resize_shape, test_labels, test_attri), batch_size=batch_size)

    return train_loader, val_loader, test_loader


def load_lidc(batch_size,
              splitnumber,
              threeD,
              resize_shape,
              args):
    """
    Generates DataLoader.
    :param batch_size: batch size of returning DataLoader
    :param splitnumber: number of split to be processed
    :param threeD: Bool, True: for cropping out volume, False: for cropping out center slice
    :param resize_shape: resize shape of twoD slice
    :return: Dataloaders of train, validation and test data
    """
    path = "LIDC_split" + str(splitnumber) + "_train.npy"
    if not os.path.isfile(path):
        print("LIDC_Split dataset nodule wise.")
        generateNoduleSplitFiles()
    path = "LIDC_img_train_split" + str(splitnumber) + ".npy"
    if not os.path.isfile(path):
        print("Create dataset for split " + str(splitnumber) + ".")
        generateDatasetforSplit(splitnumber, threeD, resize_shape)

    img_train = np.load("LIDC_img_train_split" + str(splitnumber) + ".npy")
    mask_train = np.load("LIDC_mask_train_split" + str(splitnumber) + ".npy")
    img_test = np.load("LIDC_img_test_split" + str(splitnumber) + ".npy")
    img_total_train = np.load("LIDC_img_total_train_split" + str(splitnumber) + ".npy")
    mask_test = np.load("LIDC_mask_test_split" + str(splitnumber) + ".npy")
    label_train = np.load("LIDC_label_train_split" + str(splitnumber) + ".npy")
    label_test = np.load("LIDC_label_test_split" + str(splitnumber) + ".npy")
    img_total_test = np.load("LIDC_img_total_test_split" + str(splitnumber) + ".npy")

    print("Internal structure balance")
    print(label_train[:,11].shape)
    print(label_test[:,11].shape)
    print(np.count_nonzero(np.round(label_train[:,11]) == 1))
    print(np.count_nonzero(np.round(label_test[:,11]) == 1))
    print(np.count_nonzero(np.round(label_train[:,11]) == 2))
    print(np.count_nonzero(np.round(label_test[:,11]) == 2))
    print(np.count_nonzero(np.round(label_train[:,11]) == 3))
    print(np.count_nonzero(np.round(label_test[:,11]) == 3))
    print(np.count_nonzero(np.round(label_train[:,11]) == 4))
    print(np.count_nonzero(np.round(label_test[:,11]) == 4))
    print("lobulation balance")
    print(label_train[:,19].shape)
    print(label_test[:,19].shape)
    print(np.count_nonzero(np.round(label_train[:,19]) == 1))
    print(np.count_nonzero(np.round(label_test[:,19]) == 1))
    print(np.count_nonzero(np.round(label_train[:,19]) == 2))
    print(np.count_nonzero(np.round(label_test[:,19]) == 2))
    print(np.count_nonzero(np.round(label_train[:,19]) == 3))
    print(np.count_nonzero(np.round(label_test[:,19]) == 3))
    print(np.count_nonzero(np.round(label_train[:,19]) == 4))
    print(np.count_nonzero(np.round(label_test[:,19]) == 4))
    print(np.count_nonzero(np.round(label_train[:,19]) == 5))
    print(np.count_nonzero(np.round(label_test[:,19]) == 5))

    split_train, split_val = train_test_split(torch.arange(img_train.shape[0]), test_size=0.1,
                                              stratify=label_train[:, 25])
    train_imgs = img_train[split_train]
    train_masks = mask_train[split_train]
    train_labels = label_train[split_train]
    train_imgs_total = img_total_train[split_train]
    val_imgs = img_train[split_val]
    val_masks = mask_train[split_val]
    val_labels = label_train[split_val]
    val_imgs_total = img_total_train[split_val]
    test_imgs = img_test
    test_masks = mask_test
    test_labels = label_test
    test_imgs_total = img_total_test

    fin_train_dataset = []
    counter = 0
    for i in range(train_imgs.shape[0]):
        attr_label = np.asarray([(train_labels[i, 9] - 1) / 4.0,
                                 (train_labels[i, 11] - 1) / 3.0,
                                 (train_labels[i, 13] - 1) / 5.0,
                                 (train_labels[i, 15] - 1) / 4.0,
                                 (train_labels[i, 17] - 1) / 4.0,
                                 (train_labels[i, 19] - 1) / 4.0,
                                 (train_labels[i, 21] - 1) / 4.0,
                                 (train_labels[i, 23] - 1) / 4.0])
        mal_label = np.asarray(
            distribution_label(x=[1., 2., 3., 4., 5.], mu=train_labels[i, 25], sig=train_labels[i, 26]))
        fin_train_dataset.append(
            [np.expand_dims(train_imgs[i], axis=0), np.expand_dims(train_masks[i], axis=0), attr_label, mal_label,
             counter, train_imgs_total[i]])
        counter += 1
    train_loader = torch.utils.data.DataLoader(fin_train_dataset, batch_size=batch_size, shuffle=True)

    fin_val_dataset = []
    counter = 0
    for i in range(val_imgs.shape[0]):
        attr_label = np.asarray([(val_labels[i, 9] - 1) / 4.0,
                                 (val_labels[i, 11] - 1) / 3.0,
                                 (val_labels[i, 13] - 1) / 5.0,
                                 (val_labels[i, 15] - 1) / 4.0,
                                 (val_labels[i, 17] - 1) / 4.0,
                                 (val_labels[i, 19] - 1) / 4.0,
                                 (val_labels[i, 21] - 1) / 4.0,
                                 (val_labels[i, 23] - 1) / 4.0])
        mal_label = np.asarray(distribution_label(x=[1., 2., 3., 4., 5.], mu=val_labels[i, 25], sig=val_labels[i, 26]))
        fin_val_dataset.append(
            [np.expand_dims(val_imgs[i], axis=0), np.expand_dims(val_masks[i], axis=0), attr_label, mal_label, counter, val_imgs_total[i]])
        counter += 1
    val_loader = torch.utils.data.DataLoader(fin_val_dataset, batch_size=batch_size)

    fin_test_dataset = []
    counter = 0
    for i in range(test_imgs.shape[0]):
        attr_label = np.asarray([(test_labels[i, 9] - 1) / 4.0,
                                 (test_labels[i, 11] - 1) / 3.0,
                                 (test_labels[i, 13] - 1) / 5.0,
                                 (test_labels[i, 15] - 1) / 4.0,
                                 (test_labels[i, 17] - 1) / 4.0,
                                 (test_labels[i, 19] - 1) / 4.0,
                                 (test_labels[i, 21] - 1) / 4.0,
                                 (test_labels[i, 23] - 1) / 4.0])
        mal_label = np.asarray(
            distribution_label(x=[1., 2., 3., 4., 5.], mu=test_labels[i, 25], sig=test_labels[i, 26]))
        fin_test_dataset.append(
            [np.expand_dims(test_imgs[i], axis=0), np.expand_dims(test_masks[i], axis=0), attr_label, mal_label,
             counter, test_imgs_total[i]])
        counter += 1
    test_loader = torch.utils.data.DataLoader(fin_test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


def distribution_label(x, mu, sig):
    """
    Generates distribution.
    :param x: sample points
    :param mu: mean
    :param sig: standard deviation
    :return: list of values representing distribution
    """
    if sig < .05:
        sig = .05
    d0 = (1. / np.sqrt(2 * torch.pi * sig)) * np.exp(-np.square(x[0] - mu) / (2 * sig))
    d1 = (1. / np.sqrt(2 * torch.pi * sig)) * np.exp(-np.square(x[1] - mu) / (2 * sig))
    d2 = (1. / np.sqrt(2 * torch.pi * sig)) * np.exp(-np.square(x[2] - mu) / (2 * sig))
    d3 = (1. / np.sqrt(2 * torch.pi * sig)) * np.exp(-np.square(x[3] - mu) / (2 * sig))
    d4 = (1. / np.sqrt(2 * torch.pi * sig)) * np.exp(-np.square(x[4] - mu) / (2 * sig))

    return [d0, d1, d2, d3, d4]

def createOrdinalLabel(maxclasses, classlabel):
    label = np.zeros(maxclasses - 1)
    for i in range(classlabel):
        label[i] = 1
    return label

def load_derm7pt(batch_size,
              splitnumber,
              resize_shape,
              args):
    if splitnumber>4:
        print("choose a splitnumber between 0 and 4")
        sys.exit()
    data_dir = "/path/to/derm7pt dataset/release_v0"
    image_dir = data_dir + "/images/"
    import pandas as pd

    labels_df = pd.read_csv(data_dir+"/meta/meta.csv")
    print(labels_df.keys())
    classes = labels_df['diagnosis'].tolist()
    for idx in range(len(classes)):
        if classes[idx] == 'basal cell carcinoma':
            classes[idx] = 3
        elif classes[idx].endswith('nevus'):
            classes[idx] = 0
        elif classes[idx] in ['dermatofibroma','lentigo','melanosis','miscellaneous','vascular lesion']:
            classes[idx] = 2
        elif classes[idx].startswith('melanoma'):
            classes[idx] = 4
        elif classes[idx] == 'seborrheic keratosis':
            classes[idx] = 1
    classes = np.asarray(classes)

    #use specified train val test indices
    train_indices = pd.read_csv(data_dir+"/meta/train_indexes.csv")['indexes'].tolist()
    val_indices = pd.read_csv(data_dir+"/meta/valid_indexes.csv")['indexes'].tolist()
    train_indices += val_indices
    train_classes = classes[train_indices]
    train_indices, val_indices =  train_test_split(train_indices, stratify=train_classes, test_size=0.1,random_state=4)
    test_indices = pd.read_csv(data_dir+"/meta/test_indexes.csv")['indexes'].tolist()

    print("count classes target")
    print(np.count_nonzero(classes == 0))
    print(np.count_nonzero(classes == 1))
    print(np.count_nonzero(classes == 2))
    print(np.count_nonzero(classes == 3))
    print(np.count_nonzero(classes == 4))
    attr_pn = labels_df['pigment_network'].tolist()
    for idx in range(len(attr_pn)):
        if attr_pn[idx] == 'absent':
            attr_pn[idx] = 0
        elif attr_pn[idx] == "typical":
            attr_pn[idx] = 1
        elif attr_pn[idx] == "atypical":
            attr_pn[idx] = 2
    print("count classes attr_pn")
    print(np.count_nonzero(np.asarray(attr_pn) == 0))
    print(np.count_nonzero(np.asarray(attr_pn) == 1))
    print(np.count_nonzero(np.asarray(attr_pn) == 2))
    attr_bmv = labels_df['blue_whitish_veil'].tolist()
    for idx in range(len(attr_bmv)):
        if attr_bmv[idx] == 'absent':
            attr_bmv[idx] = 0
        elif attr_bmv[idx] == "present":
            attr_bmv[idx] = 1
    print("count classes attr_bmv")
    print(np.count_nonzero(np.asarray(attr_bmv) == 0))
    print(np.count_nonzero(np.asarray(attr_bmv) == 1))
    attr_vs = labels_df['vascular_structures'].tolist()
    for idx in range(len(attr_vs)):
        if attr_vs[idx] == 'absent':
            attr_vs[idx] = 0
        elif attr_vs[idx] in ["arborizing","comma","hairpin","within regression","wreath"]:
            attr_vs[idx] = 1
        elif attr_vs[idx] in ["dotted","linear irregular"]:
            attr_vs[idx] = 2
    print("count classes attr_vs")
    print(np.count_nonzero(np.asarray(attr_vs) == 0))
    print(np.count_nonzero(np.asarray(attr_vs) == 1))
    print(np.count_nonzero(np.asarray(attr_vs) == 2))
    attr_pig = labels_df['pigmentation'].tolist()
    for idx in range(len(attr_pig)):
        if attr_pig[idx] == 'absent':
            attr_pig[idx] = 0
        elif attr_pig[idx] in ["diffuse regular","localized regular"]:
            attr_pig[idx] = 1
        elif attr_pig[idx] in ["diffuse irregular","localized irregular"]:
            attr_pig[idx] = 2
    print("count classes attr_pig")
    print(np.count_nonzero(np.asarray(attr_pig) == 0))
    print(np.count_nonzero(np.asarray(attr_pig) == 1))
    print(np.count_nonzero(np.asarray(attr_pig) == 2))
    attr_str = labels_df['streaks'].tolist()
    for idx in range(len(attr_str)):
        if attr_str[idx] == 'absent':
            attr_str[idx] = 0
        elif attr_str[idx] == "regular":
            attr_str[idx] = 1
        elif attr_str[idx] == "irregular":
            attr_str[idx] = 2
    print("count classes attr_str")
    print(np.count_nonzero(np.asarray(attr_str) == 0))
    print(np.count_nonzero(np.asarray(attr_str) == 1))
    print(np.count_nonzero(np.asarray(attr_str) == 2))
    attr_dag = labels_df['dots_and_globules'].tolist()
    for idx in range(len(attr_dag)):
        if attr_dag[idx] == 'absent':
            attr_dag[idx] = 0
        elif attr_dag[idx] == "regular":
            attr_dag[idx] = 1
        elif attr_dag[idx] == "irregular":
            attr_dag[idx] = 2
    print("count classes attr_dag")
    print(np.count_nonzero(np.asarray(attr_dag) == 0))
    print(np.count_nonzero(np.asarray(attr_dag) == 1))
    print(np.count_nonzero(np.asarray(attr_dag) == 2))
    attr_rs = labels_df['regression_structures'].tolist()
    for idx in range(len(attr_rs)):
        if attr_rs[idx] == 'absent':
            attr_rs[idx] = 0
        else:
            attr_rs[idx] = 1
    print("count classes attr_rs")
    print(np.count_nonzero(np.asarray(attr_rs) == 0))
    print(np.count_nonzero(np.asarray(attr_rs) == 1))
    attrs = np.stack([attr_pn,attr_bmv,attr_vs,attr_pig,attr_str,attr_dag,attr_rs])
    print(classes.shape)
    print(attrs.shape)
    from PIL import Image
    dataset_train_custom = []
    dataset_val_custom = []
    dataset_test_custom = []
    train_counter = 0
    val_counter = 0
    test_counter = 0
    import torchvision.transforms.v2 as tr
    transforms_ = tr.Compose([
        tr.CenterCrop(size=450)
    ])
    transforms_train = tr.Compose([
        tr.RandomHorizontalFlip(p=0.5),
        tr.RandomVerticalFlip(p=0.5)
    ])
    allimagedirs = labels_df['derm'].tolist()
    for sampleidx in range(len(classes)):
        img_path = image_dir +allimagedirs[sampleidx]
        image = Image.open(img_path)
        if sampleidx in train_indices:
            if args.base_model == "ConvNet":
                image = image.convert('L')
            image = transforms_(image)
            image = transforms_train(image)
            if args.base_model == "ConvNet":
                image = np.asarray((image).resize(resize_shape)) / 255
                image = np.expand_dims(image, axis=0)
            else:
                image = np.asarray((image).resize(resize_shape)).transpose(2, 0, 1) / 255
            dataset_train_custom.append([image, np.expand_dims(image[0],axis=0), attrs[:, sampleidx], classes[sampleidx], train_counter])
            train_counter += 1
        elif sampleidx in val_indices:
            if args.base_model == "ConvNet":
                image = image.convert('L')
            if args.base_model == "ConvNet":
                image = np.asarray((image).resize(resize_shape)) / 255
                image = np.expand_dims(image, axis=0)
            else:
                image = np.asarray(transforms_(image).resize(resize_shape)).transpose(2, 0, 1) / 255
            dataset_val_custom.append([image, np.expand_dims(image[0],axis=0), attrs[:, sampleidx], classes[sampleidx], val_counter])
            val_counter += 1
        elif sampleidx in test_indices:
            if args.base_model == "ConvNet":
                image = image.convert('L')
            if args.base_model == "ConvNet":
                image = np.asarray((image).resize(resize_shape)) / 255
                image = np.expand_dims(image, axis=0)
            else:
                image = np.asarray(transforms_(image).resize(resize_shape)).transpose(2, 0, 1) / 255
            dataset_test_custom.append([image, np.expand_dims(image[0],axis=0), attrs[:, sampleidx], classes[sampleidx], test_counter])
            test_counter += 1
    train_loader = torch.utils.data.DataLoader(dataset_train_custom, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset_val_custom, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset_test_custom, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader