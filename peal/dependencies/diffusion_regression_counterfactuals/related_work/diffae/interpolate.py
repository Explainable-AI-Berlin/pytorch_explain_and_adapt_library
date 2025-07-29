import torch
import numpy as np
import matplotlib.pyplot as plt
from templates import *
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    conf = ffhq256_autoenc()
    model = LitModel(conf)
    state = torch.load(f"checkpoints/{conf.name}/last.ckpt", map_location="cpu")
    model.load_state_dict(state["state_dict"], strict=False)
    model.ema_model.eval()
    model.ema_model.to(device)
    return model, conf


def load_data(conf):
    data = ImageDataset(
        "imgs_interpolate",
        image_size=conf.img_size,
        exts=["jpg", "JPG", "png"],
        do_augment=False,
    )
    batch = torch.stack([data[0]["img"], data[1]["img"]])
    return batch


def plot_images(batch):
    plt.imshow(batch[0].permute([1, 2, 0]) / 2 + 0.5)
    plt.show()


def encode_images(model, batch):
    cond = model.encode(batch.to(device))
    xT = model.encode_stochastic(batch.to(device), cond, T=250)
    return cond, xT


def plot_encoded_images(batch, xT):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ori = (batch + 1) / 2
    ax[0].imshow(ori[0].permute(1, 2, 0).cpu())
    ax[1].imshow(xT[0].permute(1, 2, 0).cpu())
    plt.show()


def interpolate_images(cond, xT):
    alpha = torch.tensor(np.linspace(0, 1, 10, dtype=np.float32)).to(cond.device)
    intp = cond[0][None] * (1 - alpha[:, None]) + cond[1][None] * alpha[:, None]

    def cos(a, b):
        a = a.view(-1)
        b = b.view(-1)
        a = F.normalize(a, dim=0)
        b = F.normalize(b, dim=0)
        return (a * b).sum()

    theta = torch.arccos(cos(xT[0], xT[1]))
    x_shape = xT[0].shape
    intp_x = (
        torch.sin((1 - alpha[:, None]) * theta) * xT[0].flatten(0, 2)[None]
        + torch.sin(alpha[:, None] * theta) * xT[1].flatten(0, 2)[None]
    ) / torch.sin(theta)
    intp_x = intp_x.view(-1, *x_shape)
    return intp_x, intp


def render_images(model, intp_x, intp):
    pred = model.render(intp_x, intp, T=20)
    return pred


def save_images(pred):
    # fig, ax = plt.subplots(1, 10, figsize=(5 * 10, 5))
    # for i in range(len(pred)):
    #     ax[i].imshow(pred[i].permute(1, 2, 0).cpu())
    # plt.savefig("imgs_manipulated/compare_new.png")
    torchvision.utils.save_image(pred, "imgs_manipulated/compare_new_grid.png", nrow=5)


if __name__ == "__main__":
    model, conf = load_model()
    batch = load_data(conf)
    plot_images(batch)
    cond, xT = encode_images(model, batch)
    plot_encoded_images(batch, xT)
    intp_x, intp = interpolate_images(cond, xT)
    pred = render_images(model, intp_x, intp)
    save_images(pred)
