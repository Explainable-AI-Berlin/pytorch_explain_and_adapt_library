import torch
import copy
import os
import numpy as np

from asyrp.utils.text_dic import SRC_TRG_TXT_DIC
from asyrp.asyrp_utils.set_t_edit_t_addnoise import set_t_edit_t_addnoise
from asyrp.asyrp_utils.save_image import save_image
from asyrp.asyrp_utils.get_pairs import precompute_pairs, random_noise_pairs
from asyrp.asyrp_utils.load_model import load_pretrained_model

# test
@torch.no_grad()
def run_test(runner):
    print("Running Test")

    if runner.args.num_mean_of_delta_hs:
        assert (
            runner.args.bs_train == 1
        ), "if you want to use mean, batch_size must be 1"

    if runner.args.target_image_id:
        runner.args.target_image_id = runner.args.target_image_id.split(" ")
        runner.args.target_image_id = [int(i) for i in runner.args.target_image_id]

    # Set runner.t_edit & runner.t_addnoise & return cosine similarity of attribute
    set_t_edit_t_addnoise(
        runner=runner,
        LPIPS_th=runner.args.lpips_edit_th,
        LPIPS_addnoise_th=runner.args.lpips_addnoise_th,
        return_clip_loss=False,
    )

    # ----------- Model -----------#
    model = load_pretrained_model(runner=runner)

    if runner.args.train_delta_block:
        model.setattr_layers(runner.args.get_h_num)
        print("Setattr layers")

    model = model.to(runner.device)
    model = torch.nn.DataParallel(model)

    save_name, load_dict = get_save_name(runner)

    seq_test, seq_train, seq_test_next, seq_test_edit = get_sequence(runner)

    hs_coeff, save_name_list = get_hs_coeff(runner, save_name)

    delta_h_dict = get_delta_h_dict(runner, seq_train, seq_test, load_dict, seq_test_edit, save_name_list, save_name, model)

    # ----------- Pre-compute -----------#
    print("Prepare identity latent...")
    # get xT
    if runner.args.load_random_noise:
        img_lat_pairs_dic = random_noise_pairs(
            runner,
            model,
            saved_noise=runner.args.saved_random_noise,
            save_imgs=runner.args.save_precomputed_images,
        )

    else:
        img_lat_pairs_dic = precompute_pairs(
            runner, model, runner.args.save_precomputed_images
        )

    # Unfortunately, ima_lat_pairs_dic does not match with batch_size
    x0_tensor = None
    model.eval()

    # Test set
    if runner.args.do_test:
        x_lat_tensor = None

        for step, (x0, _, x_lat) in enumerate(img_lat_pairs_dic["test"]):
            if runner.args.target_image_id:
                assert (
                    runner.args.bs_train == 1
                ), "target_image_id is only supported for batch_size == 1"
                if not step in runner.args.target_image_id:
                    continue

            if runner.args.start_image_id > step:
                continue

            if x_lat_tensor is None:
                x_lat_tensor = x_lat
                if runner.args.use_x0_tensor:
                    x0_tensor = x0
            else:
                x_lat_tensor = torch.cat((x_lat_tensor, x_lat), dim=0)
                if runner.args.use_x0_tensor:
                    x0_tensor = torch.cat((x0_tensor, x0), dim=0)
            if (step + 1) % runner.args.bs_train != 0:
                continue

            save_image(
                runner=runner,
                model=model,
                x_lat_tensor=x_lat_tensor,
                seq_inv=seq_test,
                seq_inv_next=seq_test_next,
                save_x0=runner.args.save_x0,
                save_x_origin=runner.args.save_x_origin,
                x0_tensor=x0_tensor,
                delta_h_dict=delta_h_dict,
                folder_dir=runner.args.test_image_folder,
                get_delta_hs=runner.args.num_mean_of_delta_hs,
                save_process_origin=runner.args.save_process_origin,
                save_process_delta_h=runner.args.save_process_delta_h,
                file_name=f"test_{step}_{runner.args.n_iter - 1}",
                hs_coeff=hs_coeff,
            )

            if step == runner.args.n_test_img - 1:
                break

            x_lat_tensor = None

    # Train set
    # TODO why would do_train be called while testing???
    '''if runner.args.do_train:
        for step, (x0, _, x_lat) in enumerate(img_lat_pairs_dic["train"]):
            if runner.args.target_image_id:
                assert (
                    runner.args.bs_train == 1
                ), "target_image_id is only supported for batch_size == 1"
                if not step in runner.args.target_image_id:
                    continue

            if runner.args.start_image_id > step:
                continue

            if x_lat_tensor is None:
                x_lat_tensor = x_lat
                if runner.args.use_x0_tensor:
                    x0_tensor = x0

            else:
                x_lat_tensor = torch.cat((x_lat_tensor, x_lat), dim=0)
                if runner.args.use_x0_tensor:
                    x0_tensor = torch.cat((x0_tensor, x0), dim=0)

            if (step + 1) % runner.args.bs_train != 0:
                continue

            save_image(
                runner,
                model,
                x_lat_tensor,
                seq_test,
                seq_test_next,
                save_x0=runner.args.save_x0,
                save_x_origin=runner.args.save_x_origin,
                x0_tensor=x0_tensor,
                delta_h_dict=delta_h_dict,
                folder_dir=runner.args.test_image_folder,
                get_delta_hs=runner.args.num_mean_of_delta_hs,
                save_process_origin=runner.args.save_process_origin,
                save_process_delta_h=runner.args.save_process_delta_h,
                file_name=f"train_{step}_{runner.args.n_iter - 1}",
                hs_coeff=hs_coeff,
            )

            if step == runner.args.n_train_img - 1:
                break

            # if mean_of_delta_hs is not exist,
            if step == runner.args.num_mean_of_delta_hs - 1:
                for keys in delta_h_dict.keys():
                    if delta_h_dict[keys] is None:
                        continue
                    delta_h_dict[keys] = delta_h_dict[keys] / (step + 1)

                sumation_delta_h = None
                sumation_num = 0
                for keys in delta_h_dict.keys():
                    if sumation_delta_h is None:
                        sumation_delta_h = copy.deepcopy(delta_h_dict[keys])
                        sumation_num = 1

                    else:
                        if delta_h_dict[keys] is None:
                            continue

                        sumation_delta_h += delta_h_dict[keys]
                        sumation_num += 1

                # if ignore_timesteps, only use delta_h_dict[0]
                delta_h_dict[0] = sumation_delta_h / sumation_num

                torch.save(
                    delta_h_dict,
                    f"checkpoint_latent/{exp_id}_{runner.args.n_test_step}_{runner.args.num_mean_of_delta_hs}.pth",
                )
                print(
                    f"Dict: checkpoint_latent/{exp_id}_{runner.args.n_test_step}_{runner.args.num_mean_of_delta_hs}.pth is saved."
                )

                runner.args.num_mean_of_delta_hs = 0
                print("now we use mean of delta_hs")

            x_lat_tensor = None'''


def get_delta_h_dict(runner, seq_train, seq_test, load_dict, seq_test_edit, save_name_list, save_name, model):
    # TODO is this only for logging?
    # init delta_h_dict.
    delta_h_dict = {}
    for i in seq_train:
        delta_h_dict[i] = None

    # Scaling
    if runner.args.n_train_step != runner.args.n_test_step:
        if runner.args.train_delta_h:
            trained_idx = 0
            test_delta_h_dict = {}
            if runner.args.ignore_timesteps:
                test_delta_h_dict[0] = delta_h_dict[0]

            interval_seq = seq_train[1] - seq_train[0]

            if not load_dict:
                print('Create Dict!!!')
                print('Create Dict!!!')
                print('Create Dict!!!')
                for i in seq_test_edit:
                    test_delta_h_dict[i] = delta_h_dict[seq_train[trained_idx]]

                    if i > seq_train[trained_idx] - interval_seq:
                        if trained_idx < len(seq_train) - 1:
                            trained_idx += 1

                del delta_h_dict
                delta_h_dict = test_delta_h_dict

            else:
                print('Load Dict!!!')
                print('Load Dict!!!')
                print('Load Dict!!!')
                delta_h_dict = {}
                for i in seq_test:
                    delta_h_dict[i] = None

        else:
            for i in seq_test:
                delta_h_dict[i] = None

    if os.path.exists(save_name_list[0]):
        # load checkpoint
        print(f"{save_name} exists. load checkpoint")
        if runner.args.train_delta_block:
            # for convince of num_mean_of_delta_hs. I've forgotten right parameters a lot of times.
            if runner.args.num_mean_of_delta_hs and load_dict:
                runner.args.train_delta_h = True
                runner.args.train_delta_block = False
                runner.args.num_mean_of_delta_hs = 0

            # delta_block load
            else:
                for i in range(runner.args.get_h_num):
                    get_h = getattr(model.module, f"layer_{i}")
                    get_h.load_state_dict(torch.load(save_name_list[i])[f"{0}"])

        if runner.args.train_delta_h:
            saved_dict = torch.load(save_name_list[0])
            if runner.args.ignore_timesteps:  # global delta h is delta_h_dict[0]
                try:
                    delta_h_dict[0] = saved_dict[f"{0}"]

                except:
                    delta_h_dict[0] = saved_dict[0]

            else:
                for i in delta_h_dict.keys():
                    try:
                        delta_h_dict[i] = saved_dict[f"{i}"]

                    except:
                        delta_h_dict[i] = saved_dict[i]

    else:
        if runner.args.num_mean_of_delta_hs:
            print("There in no pre-computed mean of delta_hs! Now compute it...")

        else:
            print(f"checkpoint({save_name}) does not exist!")
            exit()

    return delta_h_dict


def get_hs_coeff(runner, save_name):
    #
    scaling_factor = (
        runner.args.n_train_step / runner.args.n_test_step * runner.args.hs_coeff_delta_h
    )
    # multi attribute
    # It need to be updated multiple attr cosine & t_edit & t_addnoise
    if runner.args.multiple_attr:
        multi_attr_list = runner.args.multiple_attr.split(" ")
        if runner.args.multiple_hs_coeff:
            multi_coeff_list = runner.args.multiple_hs_coeff.split(" ")
            multi_coeff_list = [float(c) for c in multi_coeff_list]
            multi_coeff_list = multi_coeff_list + [1.0] * (
                len(multi_attr_list) - len(multi_coeff_list)
            )

        else:
            multi_coeff_list = [1.0] * len(multi_attr_list)

        save_name_list = []
        max_cosine = 0
        max_attr = ""
        for attribute in multi_attr_list:
            save_name_list.append(save_name.replace("attribute", attribute))
            runner.src_txts = SRC_TRG_TXT_DIC[attribute][0]
            runner.trg_txts = SRC_TRG_TXT_DIC[attribute][1]
            cosine = set_t_edit_t_addnoise(
                runner=runner,
                LPIPS_th=runner.args.lpips_edit_th,
                LPIPS_addnoise_th=runner.args.lpips_addnoise_th,
                return_clip_loss=False,
            )
            if cosine > max_cosine:
                max_cosine = cosine
                max_attr = attribute

        print(f"Max cosine: {max_cosine}, Max attribute: {max_attr}")
        runner.src_txts = SRC_TRG_TXT_DIC[max_attr][0]
        runner.trg_txts = SRC_TRG_TXT_DIC[max_attr][1]
        set_t_edit_t_addnoise(
            runner=runner,
            LPIPS_th=runner.args.lpips_edit_th,
            LPIPS_addnoise_th=runner.args.lpips_addnoise_th,
            return_clip_loss=False,
        )

        hs_coeff = [1.0 * runner.args.hs_coeff_origin_h] + [
            1.0 / (len(multi_attr_list)) ** (0.5) * scaling_factor * coeff
            for coeff in multi_coeff_list
        ]
        hs_coeff = tuple(hs_coeff)

    else:
        save_name_list = [save_name]
        hs_coeff = (1.0 * runner.args.hs_coeff_origin_h, 1.0 * scaling_factor)

    if runner.args.delta_interpolation:
        if runner.args.multiple_attr:
            assert (
                runner.args.get_h_num == 2
            ), "delta_multiple_attr_interpolation is only supported for get_h_num == 2"
            interpolation_vals = np.linspace(
                runner.args.min_delta, runner.args.max_delta, runner.args.num_delta
            )
            interpolation_vals = interpolation_vals.tolist()
            hs_coeff = list(hs_coeff)

            hs_coeff_list = []

            for val_1 in interpolation_vals:
                for val_2 in interpolation_vals:
                    coeff_tuple = (1.0, val_1 * hs_coeff[1], val_2 * hs_coeff[2])
                    hs_coeff_list.append(coeff_tuple)

            del hs_coeff
            hs_coeff = hs_coeff_list

        else:
            interpolation_vals = np.linspace(
                runner.args.min_delta, runner.args.max_delta, runner.args.num_delta
            )
            interpolation_vals = interpolation_vals.tolist()

            hs_coeff_list = []

            for val in interpolation_vals:
                coeff_tuple = [val * elem for elem in hs_coeff]
                coeff_tuple[0] = 1.0
                hs_coeff_list.append(tuple(coeff_tuple))

            del hs_coeff
            hs_coeff = hs_coeff_list

    return hs_coeff, save_name_list


def get_save_name(runner):
    load_dict = False
    exp_id = os.path.split(runner.args.exp)[-1]
    if runner.args.load_from_checkpoint:
        # load_from_checkpoint is exp_id
        save_name = f"checkpoint/{runner.args.load_from_checkpoint}_LC_{runner.config.data.category}_t{runner.args.t_0}"
        save_name += f"_ninv{runner.args.n_inv_step}_ngen{runner.args.n_train_step}_{runner.args.n_iter - 1}.pth"

    else:
        # TODO why is this like this?
        #save_name = f"checkpoint/{exp_id}_{runner.args.n_iter - 1}.pth"
        save_name = f"{runner.args.exp}/checkpoint_{runner.args.n_iter - 1}.pth"

    if runner.args.manual_checkpoint_name:
        # manual_checkpoint_name is full name of checkpoint
        save_name = "checkpoint/" + runner.args.manual_checkpoint_name

    elif runner.args.choose_checkpoint_num:
        # choose the iter of checkpoint
        if runner.args.load_from_checkpoint:
            save_name = f"checkpoint/{runner.args.load_from_checkpoint}_LC_{runner.config.data.category}"
            save_name += f"_t{runner.args.t_0}_ninv{runner.args.n_inv_step}_ngen{runner.args.n_train_step}"
            save_name += f"_{runner.args.n_iter - 1}_{runner.args.choose_checkpoint_num}.pth"

        else:
            save_name = f"checkpoint/{exp_id}_{runner.args.n_iter - 1}_{runner.args.choose_checkpoint_num}.pth"

    # For global delta h
    if runner.args.num_mean_of_delta_hs:
        # already exist then load
        if os.path.isfile(
            f"checkpoint_latent/{exp_id}_{runner.args.n_test_step}_{runner.args.num_mean_of_delta_hs}.pth"
        ):
            save_name = f"checkpoint_latent/{exp_id}_{runner.args.n_test_step}_{runner.args.num_mean_of_delta_hs}.pth"
            load_dict = True

        else:
            # not exist then create
            load_dict = False

    return save_name, load_dict


def get_sequence(runner):
    # ----------- Get seq -----------#
    # For editing timesteps
    if runner.args.n_train_step != 0:
        seq_train = np.linspace(0, 1, runner.args.n_train_step) * runner.args.t_0
        seq_train = seq_train[seq_train >= runner.t_edit]
        seq_train = [int(s + 1e-6) for s in list(seq_train)]
        print("Uniform skip type")

    else:
        seq_train = list(range(runner.t_edit, runner.args.t_0))
        print("No skip")

    seq_train_next = [-1] + list(seq_train[:-1])

    # For sampling
    seq_test = np.linspace(0, 1, runner.args.n_test_step) * runner.args.t_0
    seq_test_edit = seq_test[seq_test >= runner.t_edit]
    seq_test_edit = [int(s + 1e-6) for s in list(seq_test_edit)]
    seq_test = [int(s + 1e-6) for s in list(seq_test)]
    seq_test_next = [-1] + list(seq_test[:-1])

    print(f"seq_train: {seq_train}")
    print(f"seq_test: {seq_test}")
    print(f"seq_test_edit: {seq_test_edit}")

    return seq_test, seq_train, seq_test_next, seq_test_edit