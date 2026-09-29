import torchvision
from tqdm.autonotebook import tqdm, trange

try:
    from pytorch_fid import fid_score
except ImportError:
    fid_score = None

try:
    import lpips
except ImportError:
    lpips = None

try:
    from ssim import ssim
except ImportError:
    ssim = None

from .renderer import *
from .config import *
from .diffusion import Sampler
from .dist_utils import *


def _get_batch_images(batch, device=None):
    if isinstance(batch, dict):
        if "img" in batch:
            imgs = batch["img"]
        elif "x" in batch:
            imgs = batch["x"]
        else:
            imgs = next(iter(batch.values()))
    elif isinstance(batch, (list, tuple)):
        imgs = batch[0]
    else:
        imgs = batch

    if device is not None and hasattr(imgs, "to"):
        imgs = imgs.to(device)
    return imgs


def make_subset_loader(
    conf: TrainConfig,
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    parallel: bool,
    drop_last=True,
):
    dataset = SubsetDataset(dataset, size=conf.eval_num_images)
    if parallel and distributed.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    else:
        sampler = None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        # with sampler, use the sample instead of this option
        shuffle=False if sampler else shuffle,
        num_workers=conf.num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


def evaluate_lpips(
    sampler: Sampler,
    model: Model,
    conf: TrainConfig,
    device,
    val_data: Dataset,
    latent_sampler: Sampler = None,
    use_inverted_noise: bool = False,
):
    """
    compare the generated images from autoencoder on validation dataset using DINOEvaluator
    """
    from peal.global_utils import DINOEvaluator
    dino_eval = DINOEvaluator(device=device)
    val_loader = make_subset_loader(
        conf,
        dataset=val_data,
        batch_size=conf.batch_size_eval,
        shuffle=False,
        parallel=True,
    )

    model.eval()
    with torch.no_grad():
        scores = {
            "lpips": [],
            "mse": [],
            "ssim": [],
            "psnr": [],
        }
        for batch in tqdm(val_loader, desc="lpips"):
            imgs = _get_batch_images(batch, device=device)

            if use_inverted_noise:
                model_kwargs = {}
                if conf.model_type.has_autoenc():
                    with torch.no_grad():
                        model_kwargs = model.encode(imgs)
                x_T = sampler.ddim_reverse_sample_loop(
                    model=model, x=imgs, clip_denoised=True, model_kwargs=model_kwargs
                )
                x_T = x_T["sample"]
            else:
                x_T = torch.randn(
                    (len(imgs), 3, conf.img_size, conf.img_size), device=device
                )

            if conf.model_type == ModelType.ddpm:
                assert use_inverted_noise
                pred_imgs = render_uncondition(
                    conf=conf,
                    model=model,
                    x_T=x_T,
                    sampler=sampler,
                    latent_sampler=latent_sampler,
                )
            else:
                pred_imgs = render_condition(
                    conf=conf,
                    model=model,
                    x_T=x_T,
                    x_start=imgs,
                    cond=None,
                    sampler=sampler,
                )

            lpips_val = dino_eval.compute_lpips(imgs, pred_imgs)
            scores["lpips"].append(torch.tensor([lpips_val], device=device))

            norm_imgs = (imgs + 1) / 2
            norm_pred_imgs = (pred_imgs + 1) / 2
            if ssim is not None:
                scores["ssim"].append(ssim(norm_imgs, norm_pred_imgs, size_average=False))
            else:
                scores["ssim"].append(torch.zeros(len(imgs), device=device))
            scores["mse"].append(
                (norm_imgs - norm_pred_imgs).pow(2).mean(dim=[1, 2, 3])
            )
            scores["psnr"].append(psnr(norm_imgs, norm_pred_imgs))

        for key in scores.keys():
            scores[key] = torch.cat(scores[key]).float()
    model.train()

    barrier()

    outs = {
        key: [
            torch.zeros(len(scores[key]), device=device)
            for i in range(get_world_size())
        ]
        for key in scores.keys()
    }
    for key in scores.keys():
        all_gather(outs[key], scores[key])

    for key in scores.keys():
        scores[key] = torch.cat(outs[key]).mean().item()

    return scores


def psnr(img1, img2):
    """
    Args:
        img1: (n, c, h, w)
    """
    v_max = 1.0
    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
    return 20 * torch.log10(v_max / torch.sqrt(mse))


def evaluate_fid(
    sampler: Sampler,
    model: Model,
    conf: TrainConfig,
    device,
    train_data: Dataset,
    val_data: Dataset,
    latent_sampler: Sampler = None,
    conds_mean=None,
    conds_std=None,
    remove_cache: bool = True,
    clip_latent_noise: bool = False,
):
    from peal.global_utils import DINOEvaluator
    dino_eval = DINOEvaluator(device=device)

    if get_rank() == 0:
        val_loader = make_subset_loader(
            conf,
            dataset=val_data,
            batch_size=conf.batch_size_eval,
            shuffle=False,
            parallel=False,
        )
        dino_eval.fit(val_loader)

    barrier()

    world_size = get_world_size()
    rank = get_rank()
    batch_size = chunk_size(conf.batch_size_eval, rank, world_size)

    generated_batches = []
    model.eval()
    with torch.no_grad():
        if conf.model_type.can_sample():
            eval_num_images = chunk_size(conf.eval_num_images, rank, world_size)
            desc = "generating images"
            for i in trange(0, eval_num_images, batch_size, desc=desc):
                batch_size = min(batch_size, eval_num_images - i)
                x_T = torch.randn(
                    (batch_size, 3, conf.img_size, conf.img_size), device=device
                )
                batch_images = render_uncondition(
                    conf=conf,
                    model=model,
                    x_T=x_T,
                    sampler=sampler,
                    latent_sampler=latent_sampler,
                    conds_mean=conds_mean,
                    conds_std=conds_std,
                )
                generated_batches.append(batch_images)
        elif conf.model_type == ModelType.autoencoder:
            if conf.train_mode.is_latent_diffusion():
                eval_num_images = chunk_size(conf.eval_num_images, rank, world_size)
                desc = "generating images"
                for i in trange(0, eval_num_images, batch_size, desc=desc):
                    batch_size = min(batch_size, eval_num_images - i)
                    x_T = torch.randn(
                        (batch_size, 3, conf.img_size, conf.img_size), device=device
                    )
                    batch_images = render_uncondition(
                        conf=conf,
                        model=model,
                        x_T=x_T,
                        sampler=sampler,
                        latent_sampler=latent_sampler,
                        conds_mean=conds_mean,
                        conds_std=conds_std,
                        clip_latent_noise=clip_latent_noise,
                    )
                    generated_batches.append(batch_images)
            else:
                train_loader = make_subset_loader(
                    conf,
                    dataset=train_data,
                    batch_size=batch_size,
                    shuffle=True,
                    parallel=True,
                )
                for batch in tqdm(train_loader, desc="generating images"):
                    imgs = _get_batch_images(batch, device=device)
                    x_T = torch.randn(
                        (len(imgs), 3, conf.img_size, conf.img_size), device=device
                    )
                    batch_images = render_condition(
                        conf=conf,
                        model=model,
                        x_T=x_T,
                        x_start=imgs,
                        cond=None,
                        sampler=sampler,
                    )
                    generated_batches.append(batch_images)
        else:
            raise NotImplementedError()
    model.train()

    barrier()

    if len(generated_batches) > 0:
        all_generated = torch.cat(generated_batches, dim=0)
    else:
        all_generated = torch.empty((0, 3, conf.img_size, conf.img_size), device=device)

    if get_rank() == 0:
        fid = dino_eval.compute_fid(all_generated)
        fid_tensor = torch.tensor(float(fid), device=device)
        broadcast(fid_tensor, 0)
    else:
        fid_tensor = torch.tensor(0.0, device=device)
        broadcast(fid_tensor, 0)

    fid = fid_tensor.item()
    print(f"dino_fid ({get_rank()}):", fid)
    return fid


def loader_to_path(loader: DataLoader, path: str, denormalize: bool):
    # not process safe!

    if not os.path.exists(path):
        os.makedirs(path)

    # write the loader to files
    i = 0
    for batch in tqdm(loader, desc="copy images"):
        imgs = _get_batch_images(batch)
        if denormalize:
            imgs = (imgs + 1) / 2
        for j in range(len(imgs)):
            torchvision.utils.save_image(imgs[j], os.path.join(path, f"{i+j}.png"))
        i += len(imgs)
