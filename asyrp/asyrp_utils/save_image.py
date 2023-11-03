import torch
import os
import time
import torchvision.utils as tvu

from tqdm import tqdm

from asyrp.utils.diffusion_utils import denoising_step

@torch.no_grad()
def save_image(
    runner,
    model,
    x_lat_tensor,
    seq_inv,
    seq_inv_next,
    save_x0=False,
    save_x_origin=False,
    save_process_delta_h=False,
    save_process_origin=False,
    x0_tensor=None,
    delta_h_dict=None,
    get_delta_hs=False,
    folder_dir="",
    file_name="",
    hs_coeff=(1.0, 1.0),
    image_space_noise_dict=None,
):
    if save_process_origin or save_process_delta_h:
        os.makedirs(os.path.join(folder_dir, file_name), exist_ok=True)

    process_num = int(save_x_origin) + (
        len(hs_coeff) if isinstance(hs_coeff, list) else 1
    )

    with tqdm(
        total=len(seq_inv) * (process_num), desc=f"Generative process"
    ) as progress_bar:
        time_s = time.time()

        x_list = []

        if save_x0:
            if x0_tensor is not None:
                x_list.append(x0_tensor.to(runner.device))

        if save_x_origin:
            # No delta h
            x = x_lat_tensor.clone().to(runner.device)

            for it, (i, j) in enumerate(
                zip(reversed((seq_inv)), reversed((seq_inv_next)))
            ):
                t = (torch.ones(runner.args.bs_train) * i).to(runner.device)
                t_next = (torch.ones(runner.args.bs_train) * j).to(runner.device)

                x, x0_t, _, _ = denoising_step(
                    x,
                    t=t,
                    t_next=t_next,
                    models=model,
                    logvars=runner.logvar,
                    sampling_type=runner.args.sample_type,
                    b=runner.betas,
                    learn_sigma=runner.learn_sigma,
                    eta=1.0
                    if (
                        runner.args.origin_process_addnoise and t[0] < runner.t_addnoise
                    )
                    else 0.0,
                )
                progress_bar.update(1)

                if save_process_origin:
                    output = torch.cat([x, x0_t], dim=0)
                    output = (output + 1) * 0.5
                    grid = tvu.make_grid(output, nrow=runner.args.bs_train, padding=1)
                    tvu.save_image(
                        grid,
                        os.path.join(
                            folder_dir, file_name, f"origin_{int(t[0].item())}.png"
                        ),
                        normalization=True,
                    )

            x_list.append(x)

        if runner.args.pass_editing:
            pass

        else:
            if not isinstance(hs_coeff, list):
                hs_coeff = [hs_coeff]

            for hs_coeff_tuple in hs_coeff:
                x = x_lat_tensor.clone().to(runner.device)

                for it, (i, j) in enumerate(
                    zip(reversed((seq_inv)), reversed((seq_inv_next)))
                ):
                    t = (torch.ones(runner.args.bs_train) * i).to(runner.device)
                    t_next = (torch.ones(runner.args.bs_train) * j).to(runner.device)

                    x, x0_t, delta_h, _ = denoising_step(
                        x,
                        t=t,
                        t_next=t_next,
                        models=model,
                        logvars=runner.logvar,
                        sampling_type=runner.args.sample_type,
                        b=runner.betas,
                        learn_sigma=runner.learn_sigma,
                        index=runner.args.get_h_num - 1
                        if not (
                            runner.args.image_space_noise_optim
                            or runner.args.image_space_noise_optim_delta_block
                        )
                        else None,
                        eta=1.0 if t[0] < runner.t_addnoise else 0.0,
                        t_edit=runner.t_edit,
                        hs_coeff=hs_coeff_tuple,
                        delta_h=None
                        if get_delta_hs
                        else delta_h_dict[0]
                        if (runner.args.ignore_timesteps and runner.args.train_delta_h)
                        else delta_h_dict[int(t[0].item())]
                        if t[0] >= runner.t_edit
                        else None,
                        ignore_timestep=runner.args.ignore_timesteps,
                        dt_lambda=runner.args.dt_lambda,
                        warigari=runner.args.warigari,
                    )
                    progress_bar.update(1)

                    if save_process_delta_h:
                        output = torch.cat([x, x0_t], dim=0)
                        output = (output + 1) * 0.5
                        grid = tvu.make_grid(
                            output, nrow=runner.args.bs_train, padding=1
                        )
                        tvu.save_image(
                            grid,
                            os.path.join(
                                folder_dir,
                                file_name,
                                f"delta_h_{int(t[0].item())}.png",
                            ),
                            normalization=True,
                        )
                    if get_delta_hs and t[0] >= runner.t_edit:
                        if delta_h_dict[t[0].item()] is None:
                            delta_h_dict[t[0].item()] = delta_h
                        else:
                            delta_h_dict[int(t[0].item())] = (
                                delta_h_dict[int(t[0].item())] + delta_h
                            )

                x_list.append(x)

    x = torch.cat(x_list, dim=0)
    x = (x + 1) * 0.5

    grid = tvu.make_grid(x, nrow=runner.args.bs_train, padding=1)

    tvu.save_image(
        grid,
        os.path.join(folder_dir, f"{file_name}_ngen{runner.args.n_train_step}.png"),
        normalization=True,
    )

    time_e = time.time()
    print(
        f"{time_e - time_s} seconds, {file_name}_ngen{runner.args.n_train_step}.png is saved"
    )
    return x[1].detach().cpu()