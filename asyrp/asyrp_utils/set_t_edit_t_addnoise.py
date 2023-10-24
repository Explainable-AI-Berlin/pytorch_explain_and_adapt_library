import torch
import os

from asyrp.losses.clip_loss import CLIPLoss

@torch.no_grad()
def set_t_edit_t_addnoise(
    runner, LPIPS_th=0.33, LPIPS_addnoise_th=0.1, return_clip_loss=False
):
    clip_loss_func = CLIPLoss(
        runner.device,
        lambda_direction=1,
        lambda_patch=0,
        lambda_global=0,
        lambda_manifold=0,
        lambda_texture=0,
        clip_model=runner.args.clip_model_name,
    )

    # ----------- Get clip cosine similarity -----------#
    print("Texts:", runner.src_txts, runner.trg_txts)
    scr_token = clip_loss_func.tokenize(runner.src_txts)
    trg_token = clip_loss_func.tokenize(runner.trg_txts)
    text_feature_scr = clip_loss_func.encode_text(scr_token)
    text_feature_trg = clip_loss_func.encode_text(trg_token)

    ## get cosine distance between features
    text_cos_distance = torch.nn.CosineSimilarity(dim=1, eps=1e-6)(
        text_feature_scr, text_feature_trg
    )
    print("text_cos_distance", text_cos_distance.item())
    cosine = text_cos_distance.item()

    # t_edit is from LPIPS(x0_t, x0)
    print("get t_edit from LPIPS distance!")
    # LPIPS_th = 0.33
    LPIPS_th = LPIPS_th * cosine

    dataset_name = str(runner.args.config).split(".")[0]
    # TODO why???
    """if dataset_name == "custom":
        dataset_name = runner.args.custom_dataset_name"""
    LPIPS_file_name = f"{dataset_name}_LPIPS_distance_x0_t.tsv"
    LPIPS_file_path = os.path.join(runner.args.exp, LPIPS_file_name)
    if not os.path.exists(LPIPS_file_path):
        if runner.args.user_defined_t_edit and (runner.args.user_defined_t_addnoise or runner.args.user_defined_t_addnoise == 0):
            runner.t_edit = runner.args.user_defined_t_edit
            runner.t_addnoise = runner.args.user_defined_t_addnoise
            print("user_defined t_edit and t_addnoise")
            print(f"t_edit: {runner.t_edit}")
            print(f"t_addnoise: {runner.t_addnoise}")
            if return_clip_loss:
                return cosine, clip_loss_func
            else:
                return cosine

        else:
            print(
                f"LPIPS file not found, get LPIPS distance first!  : {LPIPS_file_path}"
            )
            import pdb
            pdb.set_trace()
            raise ValueError

    import csv

    lpips_dict = {}
    with open(LPIPS_file_path, "r") as f:
        lines = csv.reader(f, delimiter="\t")
        for line in lines:
            lpips_dict[int(line[0])] = float(line[1])

    sorted_lpips_dict_key_list = list(lpips_dict.keys())
    sorted_lpips_dict_key_list.sort()
    if len(sorted_lpips_dict_key_list) != 1000:
        # even if not fully steps, it's okay.
        print("Warning: LPIPS file not fully steps! (But it's okay. lol)")

    if runner.args.user_defined_t_edit or runner.args.user_defined_t_edit == 0:
        # when you use user_defined_t_edit but not user_defined_t_addnoise
        t_edit = runner.args.user_defined_t_edit

    else:
        # get t_edit
        for key in sorted_lpips_dict_key_list:
            if lpips_dict[key] >= LPIPS_th:
                t_edit = key
                break

    runner.t_edit = t_edit
    print(f"t_edit: {runner.t_edit}")

    t_addnoise = None

    # t_boost is from LPIPS(xt, x0)
    if runner.args.user_defined_t_addnoise or runner.args.user_defined_t_addnoise == 0:
        # when you use user_defined_t_addnoise but not user_defined_t_edit
        t_addnoise = runner.args.user_defined_t_addnoise

    else:
        if runner.args.add_noise_from_xt:
            LPIPS_file_name = f"{dataset_name}_LPIPS_distance_x.tsv"
            LPIPS_file_path = os.path.join(runner.args.exp, LPIPS_file_name)
            if not os.path.exists(LPIPS_file_path):
                print("LPIPS file not found, get LPIPS distance first!")
                raise ValueError

            lpips_dict = {}
            with open(LPIPS_file_path, "r") as f:
                lines = csv.reader(f, delimiter="\t")
                for line in lines:
                    lpips_dict[int(line[0])] = float(line[1])

            sorted_lpips_dict_key_list = list(lpips_dict.keys())
            sorted_lpips_dict_key_list.sort()

        # get t_add_noise
        for key in sorted_lpips_dict_key_list:
            if lpips_dict[key] >= LPIPS_addnoise_th:
                t_addnoise = key
                break

    if t_addnoise is None:
        # TODO this seems sketchy!
        t_addnoise = len(sorted_lpips_dict_key_list) - 1

    runner.t_addnoise = t_addnoise
    print(f"t_addnoise: {runner.t_addnoise}")


    if return_clip_loss:
        return cosine, clip_loss_func

    else:
        return cosine