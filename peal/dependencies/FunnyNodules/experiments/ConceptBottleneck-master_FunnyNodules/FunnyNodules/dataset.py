
# import torch
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as T
#
# import numpy as np
# from PIL import Image
# import math
# from scipy.ndimage import gaussian_filter

# def generate_nodule(size_px=224, roundness=1, spiculation=1, edge_sharpness=1,
#                     size_attr=3, intensity=3, central_structure=0, seed=None):
#     """
#     Erzeugt ein einzelnes synthetisches Nodule-Bild (numpy array, dtype=uint8).
#     Attribute:
#       - roundness: 1 (rund) .. 5 (oval)
#       - spiculation: 1 (glatt) .. 5 (stark gezackt)
#       - edge_sharpness: 1 (sehr scharf) .. 5 (weich)
#       - size_attr: 1 (klein) .. 5 (groß)
#       - intensity: 1 (dunkel) .. 5 (hell)
#       - central_structure: 0 (ohne) / 1 (mit Struktur in der Mitte)
#     """
#     rnd = np.random.RandomState(seed)
#
#     # Parameter normalisieren
#     aspect = 1.0 - (roundness - 1) * (0.55/4.0)             # Achsenverhältnis
#     spike_strength = (spiculation - 1) * (0.9/4.0) * 3         # Spikulation
#     blur_sigma = (edge_sharpness - 1) * (3.0/4.0)           # Kantenschärfe
#     size_frac = 0.08 + (size_attr - 1) * (0.35/4.0) / 7         # Größe
#     inten = 0.15 + (intensity - 1) * (0.7/4.0)              # Helligkeit
#
#     # Koordinatenraster
#     h = w = size_px
#     yy, xx = np.mgrid[:h, :w]
#     cy = h/2 + rnd.uniform(-0.05*h,0.05*h)
#     cx = w/2 + rnd.uniform(-0.05*w,0.05*w)
#     x, y = xx - cx, yy - cy
#
#     # Rotation
#     theta = rnd.uniform(0, 2*np.pi)
#     c_t, s_t = math.cos(theta), math.sin(theta)
#     xr, yr = c_t * x + s_t * y, -s_t * x + c_t * y
#
#     ang = np.arctan2(yr, xr)
#     base_r = size_frac * min(h,w)
#
#     # Spikulation
#     n_spikes = int(3 + spike_strength * 9)
#     spike_centers = np.linspace(-np.pi, np.pi, n_spikes, endpoint=False) \
#                     + rnd.normal(0, 0.4, size=n_spikes)
#     ang_pert = np.zeros_like(ang)
#     sigma_ang = 0.4 - 0.06*spiculation
#     for ac in spike_centers:
#         diff = np.angle(np.exp(1j*(ang - ac)))
#         ang_pert += np.exp(-0.5*(diff/sigma_ang)**2)
#     ang_pert = ang_pert / (ang_pert.max() + 1e-9)
#
#     radial = base_r * (1 + spike_strength * 0.6 * ang_pert)
#
#     # Elliptizität
#     xr_scaled = xr * aspect
#     r_ell = np.hypot(xr_scaled, yr)
#     mask = (r_ell <= radial).astype(float)
#
#     # Kanten weichzeichnen
#     if blur_sigma > 0.001:
#         mask = gaussian_filter(mask, sigma=blur_sigma * (size_px/64.0))
#
#     binary_mask = (mask > 0.5).astype(np.uint8)
#
#     if central_structure:
#         inner_r = base_r * 0.15 * (0.8 + rnd.rand() * 0.8) * 2
#         coords = np.argwhere(mask > 0.5)
#         for _ in range(1000):
#             center_y, center_x = coords[rnd.randint(len(coords))]
#             dist_center = np.hypot(xx - center_x, yy - center_y)
#             central = (dist_center <= inner_r).astype(float)
#             # Prüfen, ob dieser Kreis komplett in der Maske liegt
#             if np.all(mask[central > 0.5] > 0.5):
#                 texture = rnd.normal(scale=5, size=mask.shape)
#                 texture = gaussian_filter(texture, sigma=0.75)  # smooth the noise ### neu
#                 mask = np.clip(mask + 0.6 * central + 0.4 * texture * central, 0, 1)
#                 break
#         # Falls kein gültiger Punkt gefunden → einfach keine Struktur einfügen
#
#     # Hintergrund & Bild
#     background = 0.05 + rnd.normal(scale=0.02, size=mask.shape)
#     img = np.clip(background + inten * mask, 0.0, 1.0)
#     img += rnd.normal(scale=0.02, size=mask.shape)  # medizinisches Rauschen
#     img = np.clip(img, 0, 1)
#
#     return (img*255).astype(np.uint8), binary_mask

# class OnTheFlyNoduleDataset(Dataset):
#     def __init__(self, num_samples, img_size=224, transform=None, seed=42):
#         self.num_samples = num_samples
#         self.img_size = img_size
#         self.transform = transform
#         self.base_seed = seed
#
#     def __len__(self):
#         return self.num_samples
#
#     def __getitem__(self, idx):
#         # Use idx-based seed → deterministic per sample
#         rng = np.random.RandomState(self.base_seed + idx)
#
#         # Random attributes (but reproducible)
#         roundness = rng.randint(1, 6)
#         spic = rng.randint(1, 6)
#         cs = rng.randint(0, 2)
#         size_attr = rng.randint(1, 6)
#         edge_sharp = rng.randint(1, 6)
#         intensity = rng.randint(1, 6)
#
#         # Generate image
#         img, mask = generate_nodule(self.img_size, roundness, spic,
#                               edge_sharp, size_attr, intensity, cs, seed=self.base_seed+idx)
#
#         # Convert to tensor
#         img = Image.fromarray(img).convert("L")
#         if self.transform:
#             img = self.transform(img)
#         else:
#             img = T.ToTensor()(img)
#         mask = torch.from_numpy(mask).unsqueeze(0).float()
#
#
#         # Compute target (same as before)
#         score = 0
#         if cs == 0:  ### neu
#             if roundness >= 4: score += 2  ### neu
#             elif roundness <= 2: score -= 2  ### neu
#         else:  ### neu
#             if roundness >= 4: score -= 2  ### neu
#             elif roundness <= 2: score += 2  ### neu
#         # if roundness >= 4: score += 2  ### alt
#         # elif roundness <= 2: score -= 1 ### alt
#         if spic >= 4: score += 2
#         # elif spic <= 2: score -= 1 ### alt
#         elif spic <= 2: score -= 2  ### neu
#         if edge_sharp >= 4: score -= 2  ### neu
#         elif edge_sharp <= 2: score += 2  ### neu
#         # if edge_sharp >= 4: score += 1 ### alt
#         # elif edge_sharp <= 2: score -= 1 ### alt
#         if size_attr >= 4: score += 2
#         elif size_attr <= 2: score -= 2  ### neu
#         # elif size_attr <= 2: score -= 1 ### alt
#         if intensity == 5: score -= 1
#         elif intensity <= 2: score += 1
#         # if cs == 1: score += 2 ### alt
#
#         if score <= -1: target = 1
#         elif score == 0: target = 2
#         elif score in [1, 2]: target = 3
#         elif score in [3, 4]: target = 4
#         else: target = 5
#
#         attrs = {
#             "roundness": (roundness-1)/4,
#             "spiculation": (spic-1)/4,
#             "edge_sharpness": (edge_sharp-1)/4,
#             "size_attr": (size_attr-1)/4,
#             "intensity": (intensity-1)/4,
#             "central_structure": cs
#         }
#
#         attr_tensor = torch.tensor(list(attrs.values()), dtype=torch.float32)
#         target = torch.tensor(target-1)#/4
#
#         return img, mask, attr_tensor, target, torch.tensor(idx), img.clone(), np.ones(6)

# def create_dataloader(num_samples: int, batch_size: int = 4, img_size: int = 224):
#     transform = T.Compose([
#         T.Resize((img_size, img_size)),
#         T.ToTensor()
#     ])
#     dataset = OnTheFlyNoduleDataset(num_samples=num_samples,
#                                     img_size=img_size,
#                                     transform=transform,
#                                     seed=1234)  # fixed seed
#     dataloader = DataLoader(dataset, batch_size=batch_size,
#                             shuffle=True, num_workers=4)
#     return dataloader

def create_FunnyNodule(train_size, val_size, test_size, img_size, batch_size,
                                    C_cols,
                                    y_cols,
                                    zscore_C,
                                    zscore_Y,
                                    data_proportion,
                                    shuffle_Cs,
                                    merge_klg_01,
                                    max_horizontal_translation,
                                    max_vertical_translation,
                                    augment=None,
                                    sampling_strategy=None,
                                    sampling_args=None,
                                    C_hat_path=None,
                                    use_small_subset=False,
                                    downsample_fraction=None):
    train_loader = create_dataloader(num_samples=train_size, batch_size=8, img_size=img_size)
    val_loader = create_dataloader(num_samples=val_size, batch_size=8, img_size=img_size)
    test_loader = create_dataloader(num_samples=test_size, batch_size=8, img_size=img_size)
    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    dataset_sizes = {'train': len(train_loader.dataset), 'val': len(val_loader.dataset), 'test': len(test_loader.dataset)}

    return dataloaders, dataset_sizes


# class AttributeVariationDataset(Dataset):
#     """
#     Dataset that generates multiple images of the *same sample* (same seed),
#     varying only one selected attribute.
#     """
#     def __init__(self, attribute_name: str, base_seed: int = 42, img_size: int = 224, fix_all_but_r_and_cs=False, transform=None):
#         self.attribute_name = attribute_name
#         self.base_seed = base_seed
#         self.img_size = img_size
#         self.transform = transform
#         assert attribute_name in [
#             "roundness", "spiculation", "edge_sharpness",
#             "size_attr", "intensity", "central_structure"
#         ], f"Invalid attribute '{attribute_name}'"
#
#         # number of variations
#         self.num_variations = 5 if attribute_name != "central_structure" else 2
#
#         # draw all other attributes once from base_seed
#         rng = np.random.RandomState(base_seed)
#         self.base_attrs = {
#             "roundness": rng.randint(1, 6),
#             "spiculation": rng.randint(1, 6),
#             "edge_sharpness": rng.randint(1, 6),
#             "size_attr": rng.randint(1, 6),
#             "intensity": rng.randint(1, 6),
#             "central_structure": rng.randint(0, 2)
#         }
#         if fix_all_but_r_and_cs:
#             self.base_attrs = {
#                 "roundness": rng.randint(1, 6),
#                 "spiculation": 3,
#                 "edge_sharpness": 3,
#                 "size_attr": 3,
#                 "intensity": 3,
#                 "central_structure": rng.randint(0, 2)
#             }
#
#     def __len__(self):
#         return self.num_variations
#
#     def __getitem__(self, idx):
#         # copy base attributes
#         attrs = self.base_attrs.copy()
#
#         # vary only selected attribute
#         if self.attribute_name == "central_structure":
#             attrs[self.attribute_name] = idx  # 0 or 1
#         else:
#             attrs[self.attribute_name] = idx + 1  # 1..5
#
#         # generate with same base seed → same noise + spatial structure
#         img, mask = generate_nodule(
#             self.img_size,
#             attrs["roundness"],
#             attrs["spiculation"],
#             attrs["edge_sharpness"],
#             attrs["size_attr"],
#             attrs["intensity"],
#             attrs["central_structure"],
#             seed=self.base_seed
#         )
#
#         # convert image to tensor
#         img = Image.fromarray(img).convert("L")
#         if self.transform:
#             img = self.transform(img)
#         else:
#             img = T.ToTensor()(img)
#         mask = torch.from_numpy(mask).unsqueeze(0).float()
#
#         # compute target (same logic as in your main dataset)
#         score = 0
#         cs = attrs["central_structure"]
#         r, s, e, sz, inten = attrs["roundness"], attrs["spiculation"], attrs["edge_sharpness"], attrs["size_attr"], attrs["intensity"]
#
#         if cs == 0:
#             if r >= 4: score += 2
#             elif r <= 2: score -= 2
#         else:
#             if r >= 4: score -= 2
#             elif r <= 2: score += 2
#         if s >= 4: score += 2
#         elif s <= 2: score -= 2
#         if e >= 4: score -= 2
#         elif e <= 2: score += 2
#         if sz >= 4: score += 2
#         elif sz <= 2: score -= 2
#         if inten == 5: score -= 1
#         elif inten <= 2: score += 1
#
#         if score <= -1: target = 1
#         elif score == 0: target = 2
#         elif score in [1, 2]: target = 3
#         elif score in [3, 4]: target = 4
#         else: target = 5
#
#         # normalize attributes (as in your dataset)
#         attr_tensor = torch.tensor([
#             (attrs["roundness"] - 1) / 4,
#             (attrs["spiculation"] - 1) / 4,
#             (attrs["edge_sharpness"] - 1) / 4,
#             (attrs["size_attr"] - 1) / 4,
#             (attrs["intensity"] - 1) / 4,
#             float(attrs["central_structure"])
#         ], dtype=torch.float32)
#
#         # target = (target - 1) / 4
#         target = torch.tensor(target - 1)#/4
#
#         # return img, mask, attr_tensor, target, torch.tensor(idx)
#         return img, mask, attr_tensor, target, torch.tensor(idx), img.clone(), np.ones(6)


# def create_attribute_dataloader(attribute_name: str, base_seed: int = 42,
#                                 batch_size: int = 4, img_size: int = 224, fix_all_but_r_and_cs=False):
#     """
#     Creates a dataloader where all samples are the same except for one varied attribute.
#     """
#     transform = T.Compose([
#         T.Resize((img_size, img_size)),
#         T.ToTensor()
#     ])
#
#     dataset = AttributeVariationDataset(
#         attribute_name=attribute_name,
#         base_seed=base_seed,
#         img_size=img_size,
#         transform=transform,
#         fix_all_but_r_and_cs=fix_all_but_r_and_cs
#     )
#
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
#     return dataloader



#######################################################################
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import cv2
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion

def generate_nodule(size_px=224, roundness=1, spiculation=1, edge_sharpness=1,
                    size_attr=3, intensity=3, internal_structure=0, seed=None):
    """
    Generates a synthetic pulmonary nodule (uint8 image) of size `size_px`×`size_px`.

    Attributes:
      - roundness: 1 (round) .. 5 (oval)
      - spiculation: 1 (none) .. 5 (marked)
      - edge_sharpness: 1 (sharp) .. 5 (soft)
      - size_attr: 1 (small) .. 5 (big)
      - intensity: 1 (dark) .. 5 (bright)
      - internal_structure: 0 (absent) / 1 (present)
    """

    rnd = np.random.RandomState(seed)

    # ----- ATTRIBUTE PARAMETERS -----
    aspect = 1.0 - (roundness - 1) * (0.3/4.0)
    spike_strength = (spiculation - 1) * (0.9/4.0) * 3
    blur_sigma = (edge_sharpness - 1) * (3.0/4.0)

    base_size_frac = 0.18
    inten = 0.15 + (intensity - 1) * (0.7/4.0)

    # ----- CANONICAL CANVAS -----
    h = w = size_px
    yy, xx = np.mgrid[:h, :w]
    cy = h / 2
    cx = w / 2
    x, y = xx - cx, yy - cy

    # ----- Rotation -----
    theta = rnd.uniform(0, 2*np.pi)
    c_t, s_t = math.cos(theta), math.sin(theta)
    xr = c_t * x + s_t * y
    yr = -s_t * x + c_t * y

    ang = np.arctan2(yr, xr)
    base_r = base_size_frac * min(h, w)

    # ----- SPICULATION -----
    n_spikes = int(3 + spike_strength * 9)
    spike_centers = np.linspace(-np.pi, np.pi, n_spikes, endpoint=False) \
                    + rnd.normal(0, 0.4, size=n_spikes)

    ang_pert = np.zeros_like(ang)
    sigma_ang = 0.4 - 0.06 * spiculation

    for ac in spike_centers:
        diff = np.angle(np.exp(1j * (ang - ac)))
        ang_pert += np.exp(-0.5 * (diff / sigma_ang)**2)

    ang_pert = ang_pert / (ang_pert.max() + 1e-9)

    # limit spike protrusion
    max_spike_factor = 1.5
    radial = base_r * np.clip(1 + 0.25 * spike_strength * ang_pert,
                              1.0, max_spike_factor)

    # ----- SHAPE / ROUNDNESS -----
    xr_scaled = xr * aspect
    yr_scaled = yr / aspect
    r_ell = np.hypot(xr_scaled, yr_scaled)

    mask = (r_ell <= radial).astype(float)
    mask_spiky = mask.copy()
    mask_smooth = (r_ell <= base_r).astype(float)

    # ----- EDGE SOFTNESS -----
    if blur_sigma > 0.001:
        mask_spiky = gaussian_filter(
            mask_spiky, sigma=blur_sigma * (size_px / 64.0)
        )

    binary_mask = (mask_spiky > 0.5).astype(np.uint8)

    # ----- ATTRIBUTE ROIs (IN CANONICAL SIZE) -----
    dist = np.hypot(xr_scaled, yr_scaled)
    inner_radius = base_r * 0.7
    outer_radius = base_r * 1.3
    roi_roundness = 1 - np.abs(dist - base_r) / (outer_radius - inner_radius)
    roi_roundness = gaussian_filter(np.clip(roi_roundness, 0, 1), sigma=1.0)

    roi_spiculation = (np.clip(mask_spiky - mask_smooth, 0, 1) > 0.1).astype(np.uint8)

    edge_width = int(np.clip(edge_sharpness * 1.2, 2, 6))
    edge_dil = binary_dilation(binary_mask, iterations=edge_width)
    edge_ero = binary_erosion(binary_mask, iterations=edge_width)
    roi_edge = np.logical_xor(edge_dil, edge_ero).astype(np.uint8)

    roi_size = binary_mask.copy()
    roi_intensity = binary_mask.copy()
    roi_internal = np.zeros_like(binary_mask)

    # ----- INTERNAL TEXTURE -----
    internal_texture = np.zeros_like(mask_spiky)
    if internal_structure:
        inner_r = base_r * 0.15 * (0.8 + rnd.rand() * 0.8) * 2
        coords = np.argwhere(mask > 0.5)
        for _ in range(1000):
            center_y, center_x = coords[rnd.randint(len(coords))]
            dist_center = np.hypot(xx - center_x, yy - center_y)
            central = (dist_center <= inner_r).astype(float)
            if np.all(mask[central > 0.5] > 0.5):
                texture = gaussian_filter(rnd.normal(scale=5, size=mask.shape),
                                          sigma=0.75)
                internal_texture = (0.6 * central + 0.4 * texture * central)
                region = central > 0.5
                if region.any():
                    m = internal_texture[region].mean()
                    internal_texture = np.clip(internal_texture - m, -0.5, 0.5)
                roi_internal = (central > 0.2).astype(np.uint8)
                break

    # ----- IMAGE SYNTHESIS -----
    background = 0.05 + rnd.normal(scale=0.01, size=mask_spiky.shape)

    img = np.clip(background + inten * mask_spiky, 0, 1)

    if internal_structure:
        region = roi_internal.astype(bool)  # internal structure mask
        hypo_texture = np.clip(internal_texture, -1, 0)  # only negative values
        # add hypo-dense texture on top
        img[region] = np.clip(img[region] + hypo_texture[region], 0, 1)

    img += rnd.normal(scale=0.01, size=mask_spiky.shape)
    img = np.clip(img, 0, 1)

    # ----- POST-SCALING -----
    scale = np.interp(size_attr, [1, 5], [0.5, 1.0])
    new_px = int(size_px * scale)
    new_px = max(10, min(new_px, size_px))  # safety
    def scale_nn(arr):
        return cv2.resize(arr.astype(np.uint8), (new_px, new_px),
                          interpolation=cv2.INTER_NEAREST)
    def scale_lin(arr):
        return cv2.resize(arr.astype(np.float32), (new_px, new_px),
                          interpolation=cv2.INTER_LINEAR)

    img_s = scale_lin(img)
    mask_s = scale_nn(binary_mask)

    attribute_ROIs = {
        "roundness": scale_nn(roi_roundness),
        "spiculation": scale_nn(roi_spiculation),
        "edge_sharpness": scale_nn(roi_edge),
        "size_attr": scale_nn(roi_size),
        "intensity": scale_nn(roi_intensity),
        "internal_structure": scale_nn(roi_internal)
    }

    canvas_img = 0.05 + rnd.normal(scale=0.01, size=(size_px, size_px)).astype(np.float32)
    canvas_mask = np.zeros_like(canvas_img, dtype=np.uint8)
    canvas_rois = {k: np.zeros_like(canvas_mask) for k in attribute_ROIs}

    start = (size_px - new_px) // 2
    end = start + new_px

    canvas_img[start:end, start:end] = img_s
    canvas_mask[start:end, start:end] = mask_s

    for k in attribute_ROIs:
        canvas_rois[k][start:end, start:end] = attribute_ROIs[k]

    return (canvas_img * 255).astype(np.uint8), canvas_mask, canvas_rois

def visualize_nodule(img, binary_mask, rois, label_values=None, figsize=(15, 3)):
    """
    Visualize the synthetic nodule image, its binary mask, and each attribute ROI.
    """
    n_rois = len(rois)
    cols = 2 + n_rois
    rows = 1

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("image")
    axes[0].axis("off")

    axes[1].imshow(binary_mask, cmap='gray')
    axes[1].set_title("mask")
    axes[1].axis("off")

    for i, (name, roi) in enumerate(rois.items(), start=2):
        axes[i].imshow(binary_mask, cmap='gray')

        overlay = np.zeros((*roi.shape, 4))
        overlay[..., 0] = 1.0
        overlay[..., 3] = roi * 0.5

        axes[i].imshow(overlay)
        if label_values and name in label_values:
            if name=="target":
                break
            axes[i].set_title(name+": "+str(label_values[name]))
        axes[i].axis("off")
    for j in range(2 + n_rois, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    fig.suptitle("target: "+str(label_values["target"]))
    plt.show()


class NoduleDataset(Dataset):
    def __init__(self, num_samples, img_size=224, transform=None, seed=42):
        self.num_samples = num_samples
        self.img_size = img_size
        self.transform = transform
        self.base_seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rng = np.random.RandomState(self.base_seed + idx)

        # Random attributes
        roundness = rng.randint(1, 6)
        spiculation = rng.randint(1, 6)
        size_attr = rng.randint(1, 6)
        edge_sharpness = rng.randint(1, 6)
        intensity = rng.randint(1, 6)
        internal_structure = rng.randint(0, 2)

        # Generate image
        img, mask, attribute_ROIs = generate_nodule(self.img_size, roundness, spiculation,
                              edge_sharpness, size_attr, intensity, internal_structure, seed=self.base_seed+idx)

        img = Image.fromarray(img).convert("L")
        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        target = calculate_target(roundness, spiculation, edge_sharpness, size_attr, intensity, internal_structure)

        attrs = {
            "roundness": (roundness-1)/4,
            "spiculation": (spiculation-1)/4,
            "edge_sharpness": (edge_sharpness-1)/4,
            "size_attr": (size_attr-1)/4,
            "intensity": (intensity-1)/4,
            "internal_structure": internal_structure
        }

        attr_tensor = torch.tensor(list(attrs.values()), dtype=torch.float32)
        target = (target-1)#/4

        return img, mask, attr_tensor, target, torch.tensor(idx), img.clone(), np.ones(6)

def calculate_target(roundness, spiculation, edge_sharpness, size_attr, intensity, internal_structure):
    score = 0
    if internal_structure == 0:
        if roundness >= 4:
            score += 2
        elif roundness <= 2:
            score -= 2
    else:
        if roundness >= 4:
            score -= 2
        elif roundness <= 2:
            score += 2
    if spiculation >= 4:
        score += 2
    elif spiculation <= 2:
        score -= 2
    if edge_sharpness >= 4:
        score -= 2
    elif edge_sharpness <= 2:
        score += 2
    if size_attr >= 4:
        score += 2
    elif size_attr <= 2:
        score -= 2
    if intensity == 5:
        score -= 1
    elif intensity <= 2:
        score += 1

    if score <= -1:
        target = 1
    elif score == 0:
        target = 2
    elif score in [1, 2]:
        target = 3
    elif score in [3, 4]:
        target = 4
    else:
        target = 5
    return target

def create_dataloader(num_samples: int, batch_size: int = 4, img_size: int = 224):
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor()
    ])
    dataset = NoduleDataset(num_samples=num_samples,
                                    img_size=img_size,
                                    transform=transform,
                                    seed=1234)  # fixed seed
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)
    return dataloader















from torch.utils.data import Dataset, DataLoader

class AttributeVariationDataset(Dataset):
    """
    Dataset that generates multiple images of the *same sample* (same seed),
    varying only one selected attribute.
    """
    def __init__(self, attribute_name: str, base_seed: int = 42, img_size: int = 224, fix_all_but_r_and_is=False, transform=None):
        self.attribute_name = attribute_name
        self.base_seed = base_seed
        self.img_size = img_size
        self.transform = transform
        assert attribute_name in [
            "roundness", "spiculation", "edge_sharpness",
            "size_attr", "intensity", "internal_structure"
        ], f"Invalid attribute '{attribute_name}'"

        self.num_variations = 5 if attribute_name != "internal_structure" else 2
        rng = np.random.RandomState(base_seed)
        self.base_attrs = {
            "roundness": rng.randint(1, 6),
            "spiculation": rng.randint(1, 6),
            "edge_sharpness": rng.randint(1, 6),
            "size_attr": rng.randint(1, 6),
            "intensity": rng.randint(1, 6),
            "internal_structure": rng.randint(0, 2)
        }
        if fix_all_but_r_and_is:
            self.base_attrs = {
                "roundness": rng.randint(1, 6),
                "spiculation": 3,
                "edge_sharpness": 3,
                "size_attr": 3,
                "intensity": 3,
                "internal_structure": rng.randint(0, 2)
            }

    def __len__(self):
        return self.num_variations

    def __getitem__(self, idx):
        attrs = self.base_attrs.copy()
        if self.attribute_name == "internal_structure":
            attrs[self.attribute_name] = idx
        else:
            attrs[self.attribute_name] = idx + 1

        img, mask, attribute_ROIs = generate_nodule(
            self.img_size,
            attrs["roundness"],
            attrs["spiculation"],
            attrs["edge_sharpness"],
            attrs["size_attr"],
            attrs["intensity"],
            attrs["internal_structure"],
            seed=self.base_seed
        )

        img = Image.fromarray(img).convert("L")
        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        target = calculate_target(attrs["roundness"], attrs["spiculation"], attrs["edge_sharpness"],
                                  attrs["size_attr"], attrs["intensity"], attrs["internal_structure"])

        attr_tensor = torch.tensor([
            (attrs["roundness"] - 1) / 4,
            (attrs["spiculation"] - 1) / 4,
            (attrs["edge_sharpness"] - 1) / 4,
            (attrs["size_attr"] - 1) / 4,
            (attrs["intensity"] - 1) / 4,
            float(attrs["internal_structure"])
        ], dtype=torch.float32)
        target = (target - 1) #/ 4

        return img, mask, attr_tensor, target, torch.tensor(idx), img.clone(), np.ones(6)


def create_attribute_dataloader(attribute_name: str, base_seed: int = 42,
                                batch_size: int = 4, img_size: int = 224, fix_all_but_r_and_is=False):
    """
    Creates a dataloader where all samples are the same except for one varied attribute.
    """
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor()
    ])

    dataset = AttributeVariationDataset(
        attribute_name=attribute_name,
        base_seed=base_seed,
        img_size=img_size,
        transform=transform,
        fix_all_but_r_and_is=fix_all_but_r_and_is
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return dataloader


