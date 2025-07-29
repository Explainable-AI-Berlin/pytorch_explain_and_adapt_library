import os
from typing import Union
import torchvision
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from diff_cf_ir.counterfactuals import CFResult
from diff_cf_ir.metrics import get_regr_confidence

import matplotlib

matplotlib.use("Agg")
matplotlib.rc("text")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_default_transforms(size: int):
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
            torchvision.transforms.Resize((size, size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x.unsqueeze(0)),
        ]
    )


class DiffeoCF:
    DIST_L1 = "l1"
    DIST_L2 = "l2"

    def __init__(
        self,
        gmodel: torch.nn.Module,
        rmodel: torch.nn.Module,
        data_shape: tuple[int, int, int],
        result_dir: str,
        dist: Union[str, None] = None,
        optimizer: str = "adam",
        dist_type: str = "latent",
        dist_factor: float = 0,
    ):
        self.gmodel = gmodel
        self.rmodel = rmodel
        self.data_shape = data_shape

        self.result_dir = result_dir
        self.steps_dir = os.path.join(result_dir, "steps")
        self.current_image: str = ""  # For intermediate images

        self.loss_fn = nn.MSELoss()

        self.dist = dist
        self.dist_type = dist_type
        self.dist_factor = dist_factor
        # if dist is not None:
        #     self.dist_factor = 0.001 if "l1" in dist else 0.1  # Like ACE

        #     if self.dist_type == "pixel":
        #         self.dist_factor *= 0.01  # Otherwise it will not work at all

        self.optimizer = optimizer

    def get_optimizer(self, params, lr) -> Optimizer:
        if self.optimizer == "adam":
            return torch.optim.Adam(params=params, lr=lr)
        elif self.optimizer == "sgd":
            return torch.optim.SGD(params=params, lr=lr)
        elif self.optimizer == "sgd_momentum":
            return torch.optim.SGD(params=params, lr=lr, momentum=0.9)
        else:
            raise NotImplementedError(f"Optimizer not implemented: {self.optimizer}")

    def adv_attack(
        self,
        x: torch.Tensor,
        attack_style: str,
        num_steps: int,
        lr: float,
        target: float,
        confidence_threshold: float,
        image_path: str,
    ) -> CFResult:
        """
        prepare adversarial attack in X or Z
        run attack
        save resulting adversarial example/counterfactual
        """
        # load image
        x = x.requires_grad_(False).to(device)

        # define parameters that will be optimized
        params = []
        if attack_style == "z":
            # define z as params for derivative wrt to z
            z = self.gmodel.encode(x)
            z = [z_i.detach() for z_i in z] if isinstance(z, list) else z.detach()
            x_org = x.detach().clone()
            z_org = [z_i.clone() for z_i in z] if isinstance(z, list) else z.clone()

            if type(z) == list:
                for z_part in z:
                    z_part.requires_grad = True
                    params.append(z_part)
            else:
                z.requires_grad = True
                params.append(z)
        else:
            # define x as params for derivative wrt x
            x_org = x.clone()
            x.requires_grad = True
            params.append(x)
            z = None

        optimizer = self.get_optimizer(params, lr)

        # intermediate images
        image_name = image_path.split("/")[-1].split(".")[0]
        self.current_image = image_name

        # run the adversarial attack
        x_prime, success, y_pred_initial, y_pred_final, step = self._run_adv_attack(
            x,
            z,
            optimizer,
            target,
            attack_style,
            confidence_threshold,
            num_steps,
        )

        if not success:
            print(
                (
                    "Warning: Maximum number of iterations exceeded! Attack did not reach target value:"
                    f"\n  Target: {target}, Init: {y_pred_initial}, Final: {y_pred_final}"
                )
            )

        # save results
        cmap_img = "jet" if self.data_shape[0] == 3 else "gray"

        return CFResult(
            image_path=image_path,
            x=x.detach().cpu(),
            x_reconstructed=self.gmodel.decode(z_org).detach().cpu(),
            x_prime=x_prime.detach().cpu(),
            y_target=target,
            y_initial_pred=y_pred_initial,
            y_final_pred=y_pred_final,
            success=success,
            steps=step,
        )

    def adv_attack_dae(
        self,
        x: torch.Tensor,
        num_steps: int,
        lr: float,
        target: float,
        stop_at: float,
        confidence_threshold: float,
        image_path: str,
    ) -> CFResult:
        """
        prepare adversarial attack in X or Z
        run attack
        save resulting adversarial example/counterfactual
        """
        # load image
        x = x.requires_grad_(False).to(device)

        # define parameters that will be optimized
        params = []

        # define z as params for derivative wrt to z
        z_sem, xT = self.gmodel.encode(x)
        z_sem.detach()
        x_org = (x.detach().cpu().clone() + 1) / 2
        z_org = z_sem.clone().detach().cpu()

        z_sem.requires_grad = True
        params.append(z_sem)

        optimizer = self.get_optimizer(params, lr)

        # intermediate images
        image_name = image_path.split("/")[-1].split(".")[0]
        self.current_image = image_name

        # run the adversarial attack
        x_prime, success, y_pred_initial, y_pred_final, step = self._run_adv_attack_dae(
            x,
            z_sem,
            xT,
            optimizer,
            target,
            confidence_threshold,
            num_steps,
            stop_at=stop_at,
            x_org=x_org,
            z_org=z_org,
        )

        if not success:
            print(
                (
                    "Warning: Maximum number of iterations exceeded! Attack did not reach target value:"
                    f"\n  Target: {target}, Init: {y_pred_initial}, Final: {y_pred_final}"
                )
            )

        with torch.no_grad():
            x_reconstr = self.gmodel.decode(z_org.to(device), xT).detach().cpu()

        return CFResult(
            image_path=image_path,
            x=x_org,
            x_reconstructed=x_reconstr,
            x_prime=x_prime.detach().cpu(),
            y_target=stop_at,
            y_initial_pred=y_pred_initial,
            y_final_pred=y_pred_final,
            success=success,
            steps=step,
        )

    def save_intermediate_img(self, x, n_iter, y_pred, save_mask: list[bool] = []):
        """
        Saves intermediate images for the attack
        """
        return
        # Temporarily disable
        # if len(x.shape) == 3:
        #     img_idx_path = os.path.join(self.steps_dir, self.current_image)

        #     y_pred_formatted = f"{y_pred.item():.3e}"
        #     img_path = os.path.join(
        #         img_idx_path, f"niter={n_iter:04d}_y={y_pred_formatted}.png"
        #     )

        #     save_img_threaded(x, img_path)
        # elif len(x.shape) == 4:
        #     for i in range(x.size(0)):
        #         if save_mask and not save_mask[i]:
        #             continue

        #         img = x[i]
        #         img_idx_path = os.path.join(self.steps_dir, self.current_image[i])

        #         y_pred_formatted = f"{y_pred[i].item():.3e}"
        #         img_path = os.path.join(
        #             img_idx_path, f"niter={n_iter:04d}_y={y_pred_formatted}.png"
        #         )
        #         save_img_threaded(img, img_path)

    def _run_adv_attack(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        optimizer: Optimizer,
        target: float,
        attack_style: str,
        confidence_threshold: float,
        num_steps: int,
    ) -> tuple[torch.Tensor, bool, float, float, int]:
        """
        run optimization process on x or z for num_steps iterations
        early stopping when confidence_threshold is reached.
        """
        target_pt = torch.Tensor([[target]]).to(x.device)

        y_pred_initial: float = -1
        y_pred_final: float = -1
        success = False

        with tqdm(range(num_steps), desc="Attacking") as progress_bar:
            for step in progress_bar:
                optimizer.zero_grad()

                if attack_style == "z":
                    # TODO Enable batches of x?
                    x = self.gmodel.decode(z)

                # assert that x is a valid image
                x.data = torch.clip(x.data, min=0.0, max=1.0)

                regression = self.rmodel(x)
                if step == 0:
                    y_pred_initial = regression.item()
                loss = self.loss_fn(regression, target_pt)

                target_confidence = get_regr_confidence(regression, target_pt).item()
                progress_bar.set_postfix(
                    rgr=regression.item(),
                    l=loss.item(),
                    confid=target_confidence,
                    max_gpu_GB=torch.cuda.max_memory_reserved() / 1e9,
                )

                self.save_intermediate_img(x[0], step, y_pred=regression)
                if target_confidence <= confidence_threshold:
                    success = True
                    break

                loss.backward()
                optimizer.step()

        y_pred_final = regression.item()
        return x, success, y_pred_initial, y_pred_final, step

    def _run_adv_attack_dae(
        self,
        x: torch.Tensor,
        z_sem: torch.Tensor,
        xT: torch.Tensor,
        optimizer: Optimizer,
        target: float,
        confidence_threshold: float,
        num_steps: int,
        stop_at: float,
        x_org: torch.Tensor,
        z_org: torch.Tensor,
    ) -> tuple[torch.Tensor, bool, float, float, int]:
        """
        run optimization process on x or z for num_steps iterations
        early stopping when confidence_threshold is reached.
        """
        target_pt = torch.Tensor([[target]]).to(x.device)
        stop_at_pt = torch.Tensor([[stop_at]]).to(x.device)

        y_pred_initial: float = -1
        y_pred_final: float = -1
        success = False

        with tqdm(
            range(num_steps), desc=f"Attacking with target={target}"
        ) as progress_bar:
            for step in progress_bar:
                optimizer.zero_grad()

                x_prime: torch.Tensor = self.gmodel.decode(z_sem, xT)
                # assert that x is a valid image
                x_prime.data = torch.clip(x_prime.data, min=0.0, max=1.0)

                loss, regression, dist = self.compute_loss(
                    target_pt, x=x_prime, x_org=x_org, z_sem=z_sem, z_org=z_org
                )

                if step == 0:
                    y_pred_initial = regression.item()

                target_confidence = get_regr_confidence(regression, stop_at_pt).item()
                progress_bar.set_postfix(
                    rgr=regression.item(),
                    dist=dist,
                    l=loss.item(),
                    confid=target_confidence,
                    max_gpu_GB=torch.cuda.max_memory_reserved() / 1e9,
                )

                self.save_intermediate_img(x_prime[0], step, y_pred=regression)
                if target_confidence <= confidence_threshold:
                    success = True
                    break

                loss.backward()
                optimizer.step()

        y_pred_final = regression.item()
        return x_prime, success, y_pred_initial, y_pred_final, step

    def get_dist(
        self,
        x: torch.Tensor,
        x_org: torch.Tensor,
        z_sem: torch.Tensor,
        z_org: torch.Tensor,
    ):
        if self.dist == self.DIST_L1:
            p = 1
        elif self.dist == self.DIST_L2:
            p = 2
        else:
            raise ValueError(f"Unknown distance type: {self.dist}")

        if self.dist_type == "latent":
            diff = z_org.to(z_sem) - z_sem
        elif self.dist_type == "pixel":
            diff = x_org.to(x) - x
        else:
            raise ValueError(f"Unknown distance mode: {self.dist_type}")

        dist = torch.linalg.vector_norm(diff, ord=p).mean()
        return dist * self.dist_factor

    def compute_loss(
        self,
        target_pt: torch.Tensor,
        x: torch.Tensor,
        x_org: torch.Tensor,
        z_sem: torch.Tensor,
        z_org: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:

        regression = self.rmodel(x)

        # TODO: WIP
        # if target_pt.item() == float("inf"):
        #     # Untargeted maximize, so we need to invert the regression value
        #     loss = -regression
        # elif target_pt.item() == float("-inf"):
        #     # Untargeted minimize, the optimizer will minimize the regression value by default
        #     loss = regression
        # else:
        #     # We have a normal target
        loss = self.loss_fn(regression, target_pt)

        if self.dist is not None:
            dist = self.get_dist(x, x_org, z_sem, z_org)
            loss += dist
            dist = dist.item()
        else:
            dist = 0
        return loss, regression, dist

    @torch.no_grad()
    def adv_attack_dae_batch(
        self,
        xs: torch.Tensor,
        targets: torch.Tensor,
        stop_ats: torch.Tensor,
        image_paths: list[str],
        confidence_threshold: float,
        num_steps: int,
        lr: float,
    ) -> list[CFResult]:
        """Runs the adversarial attack on a batch of images for DAEs.

        Does not supported untargeted attacks.

        Parameters
        ----------
        xs : torch.Tensor
            Input images
        targets : torch.Tensor
            Targets for the regression model. Can be inf or -inf for untargeted attacks.
        stop_ats : torch.Tensor
            Stops the attack once MAE(regression value, stop_at) <= confidence_threshold
        image_paths : list[str]
            Paths of the images for saving the results
        confidence_threshold : float
            Threshold for stopping the attack
        num_steps : int
            maximum number of steps
        lr : float
            Learning rate for the optimizer

        Returns
        -------
        list[CFResult]
            List of results for each image
        """

        # load image
        xs = xs.to(device)
        targets = targets.to(device)
        stop_ats = stop_ats.to(device)

        # define parameters that will be optimized
        params = []

        # define z as params for derivative wrt to z
        z_sems, xTs = self.gmodel.encode(xs)
        z_sems.detach()
        x_orgs = (xs.detach().cpu().clone() + 1) / 2
        z_orgs = z_sems.clone().detach().cpu()

        # intermediate images
        image_names = [
            image_path.split("/")[-1].split(".")[0] for image_path in image_paths
        ]
        self.current_image = image_names

        # run the adversarial attack
        x_primes, successes, y_preds_initials, y_preds_finals, steps = (
            self._run_adv_attack_dae_batch(
                xs=xs,
                z_sems=z_sems,
                xTs=xTs,
                targets=targets,
                stop_ats=stop_ats,
                confidence_threshold=confidence_threshold,
                lr=lr,
                num_steps=num_steps,
                x_orgs=x_orgs,
                z_orgs=z_orgs,
            )
        )

        with torch.no_grad():
            x_reconstructions = (
                self.gmodel.decode(z_orgs.to(device), xTs).detach().cpu()
            )

        results = []
        for i in range(len(x_primes)):
            results.append(
                CFResult(
                    image_path=image_paths[i],
                    x=x_orgs[i].unsqueeze(0),
                    x_reconstructed=x_reconstructions[i].unsqueeze(0),
                    x_prime=x_primes[i].detach().cpu().unsqueeze(0),
                    y_target=stop_ats[
                        i
                    ].item(),  # need to use the stop_ats value for targets
                    y_initial_pred=y_preds_initials[i].item(),
                    y_final_pred=y_preds_finals[i].item(),
                    success=successes[i],
                    steps=steps[i],
                )
            )

        return results

    def _run_adv_attack_dae_batch(
        self,
        xs: torch.Tensor,
        z_sems: torch.Tensor,
        xTs: torch.Tensor,
        targets: torch.Tensor,
        stop_ats: torch.Tensor,
        confidence_threshold: float,
        lr: float,
        num_steps: int,
        x_orgs: torch.Tensor,
        z_orgs: torch.Tensor,
    ) -> tuple[torch.Tensor, list[bool], torch.Tensor, torch.Tensor, list[int]]:
        """
        Run adversarial attack using DAE in batch mode.

        Parameters
        ----------
        xs : torch.Tensor
            Input tensor of original images.
        z_sems : torch.Tensor
            Latent semantic representations of the input images.
        xTs : torch.Tensor
            Noise tensors for z_sems.
        targets : torch.Tensor
            Target values for the adversarial attack.
        stop_ats : torch.Tensor
            Threshold values to stop the attack.
        confidence_threshold : float
            Confidence threshold to determine the success of the attack.
        optimizer : Optimizer
            Optimizer used for updating the model parameters.
        num_steps : int
            Number of steps to run the attack.
        x_orgs : torch.Tensor
            Original input images tensor.
        z_orgs : torch.Tensor
            Original latent semantic representations tensor.

        Returns
        -------
        tuple[torch.Tensor, list[bool], torch.Tensor, torch.Tensor, list[int]]
            A tuple containing:
            - x_prime (torch.Tensor): The adversarially perturbed images.
            - success (list[bool]): List indicating whether the attack was successful.
            - y_preds_initial (list[float]): Initial prediction values.
            - y_preds_final (list[float]): Final prediction values.
            - steps (list[int]): The steps at which the attacks were stopped.
        """
        y_preds_initial: torch.Tensor = None
        y_preds_final: torch.Tensor = None
        confidence_thresholds: torch.Tensor = torch.Tensor([confidence_threshold]).to(
            z_sems
        )
        steps_needed = torch.zeros(xs.size(0), dtype=torch.int16)

        z_list = z_sems.split(1, dim=0)
        for z in z_list:
            z.requires_grad_()

        optimizer = self.get_optimizer(z_list, lr)

        def track_successful_attacks(
            i: int,
            confidence_thresholds: torch.Tensor,
            confidences: torch.Tensor,
            steps_needed: torch.Tensor,
        ):
            successful_attacks = (confidences <= confidence_thresholds).cpu()
            steps_needed = torch.where(
                ~successful_attacks.view(-1),
                i,
                steps_needed,
            )
            # Update the z_sems that have reached the confidence threshold
            for i, success in enumerate(successful_attacks):
                if success.item():
                    z_list[i].requires_grad = False

            done = all([not z.requires_grad for z in z_list])
            return steps_needed, done

        x_primes: torch.Tensor = None
        with torch.enable_grad():
            with tqdm(range(num_steps), desc=f"Attacking Batch") as pbar:

                def update_pbar(y_hat, confidences, loss, dist):
                    pbar.set_postfix(
                        confidence_mean=confidences.mean().item(),
                        regr=[f"{val:.4f}" for val in y_hat.cpu().view(-1).tolist()],
                        regr_attacking=[1 if z.requires_grad else 0 for z in z_list],
                        dist=dist,
                        dist_percent=f"{dist / loss:.1%}",
                        max_gpu_GB=torch.cuda.max_memory_reserved() / 1e9,
                        steps_needed=steps_needed.tolist(),
                    )

                for step in pbar:
                    z_sems_current = torch.cat(z_list, dim=0)
                    x_primes: torch.Tensor = self.gmodel.decode(z_sems_current, xTs)
                    # assert that x is a valid image
                    x_primes.data = torch.clip(x_primes.data, min=0.0, max=1.0)

                    loss, regression_values, dist = self.compute_loss(
                        targets,
                        x=x_primes,
                        x_org=x_orgs,
                        z_sem=z_sems_current,
                        z_org=z_orgs,
                    )

                    if step == 0:
                        y_preds_initial = regression_values.detach().cpu()

                    confidences = get_regr_confidence(regression_values, stop_ats)

                    self.save_intermediate_img(
                        x_primes,
                        step,
                        y_pred=regression_values,
                        save_mask=[z.requires_grad for z in z_list],
                    )

                    # Handle the images that have reached the confidence threshold and update mask
                    steps_needed, done = track_successful_attacks(
                        step,
                        confidence_thresholds,
                        confidences,
                        steps_needed,
                    )
                    update_pbar(regression_values, confidences, loss.item(), dist)
                    if done:
                        break

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

        y_preds_final = regression_values.detach().cpu()
        successes: list[bool] = [not z.requires_grad for z in z_list]

        return (
            x_primes,
            successes,
            y_preds_initial,
            y_preds_final,
            steps_needed.tolist(),
        )
