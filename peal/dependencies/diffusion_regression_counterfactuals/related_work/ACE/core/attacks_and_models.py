import os

from tqdm import tqdm
import torch

from diff_cf_ir.metrics import get_regr_confidence
from diff_cf_ir.file_utils import save_img_threaded


# =======================================================
# Pytorch models
# =======================================================


class JointClassifierDDPM(torch.nn.Module):
    """
    module to compute easily the gradients when using
    a ddpm + classifier.
    Computes the following:
        1) x ---> x' = noise(x) [noisy version at t=steps]
        2) x' --> x_c = ddpm(x') [iterative denoising]
        3) x_c -> l = m(x_c) [model output]
        4) return l [returns whatever m returns]
    """

    def __init__(self, classifier, ddpm, diffusion, steps, stochastic):
        super().__init__()
        self.ddpm = ddpm
        self.steps = steps
        self.diffusion = diffusion
        self.classifier = classifier
        self.stochastic = stochastic
        self.noise_fn = torch.randn_like if stochastic else torch.zeros_like
        self.index = 0

    def forward(self, x):

        timesteps = list(range(self.steps))[::-1]

        x = (x - 0.5) / 0.5

        for idx, t in enumerate(timesteps):

            t = torch.tensor([t] * x.size(0), device=x.device)

            if idx == 0:
                x = self.diffusion.q_sample(x, t, noise=self.noise_fn(x))

            out = self.diffusion.p_mean_variance(self.ddpm, x, t, clip_denoised=True)

            x = out["mean"]

            if idx != (self.steps - 1):
                if self.stochastic:
                    x += torch.exp(0.5 * out["log_variance"]) * self.noise_fn(x)

            # self.save_noised(x, t[0].item(), self.stochastic)

        x = (x * 0.5) + 0.5

        self.index += 1

        return self.classifier(x)

    @torch.no_grad()
    def initial(self, x):
        timesteps = list(range(self.steps))[::-1]

        x = x.clone().detach()
        initial_pred = self.classifier(x)

        x = (x - 0.5) / 0.5

        for idx, t in enumerate(timesteps):

            t = torch.tensor([t] * x.size(0), device=x.device)

            if idx == 0:
                x = self.diffusion.q_sample(x, t, noise=self.noise_fn(x))

            out = self.diffusion.p_mean_variance(self.ddpm, x, t, clip_denoised=True)

            x = out["mean"]

            if idx != (self.steps - 1):
                if self.stochastic:
                    x += torch.exp(0.5 * out["log_variance"]) * self.noise_fn(x)

        x = (x * 0.5) + 0.5

        return x.detach().cpu(), initial_pred


# =======================================================
# Attack Bases
# =======================================================


class Attack:
    """
    Base Attack class. Computes basic things such as:
        - Set distance schedule.
        - Computes l2 and linf projections
        - Choses if the attack is untargetted or
          targetted (done via perturb fn)
    """

    def __init__(
        self,
        predict,
        loss_fn,
        dist_fn,
        confidence_threshold,
        steps_dir,
        eps,
        step=1 / 255,
        nb_iter=100,
        norm="linf",
        dist_schedule="none",
        binary=False,
        predictor: torch.nn.Module = None,
    ):
        """
        :param predict: classification model
        :param loss_fn: loss function
        :param dist_fn: distance function
        :param eps: attack budget
        :param step: optimization step
        :param nb_iters: number of iterations
        :param norm: ball norm
        :param dist_schedule: schedule type for the distance loss
        :param binary: flag to tell if the model is binary of multi class
        """
        self.predict = predict
        assert (
            predictor is not None
        ), "Parameter predictor must be defined for regression version."
        self.predictor: torch.nn.Module = predictor
        self.loss_fn = loss_fn
        self.dist_fn = dist_fn

        # attack opts
        self.eps = eps
        self.nb_iter = nb_iter
        self.norm = norm
        self.step = step
        assert norm in ["linf", "l2"], 'PGD norm must by "linf" or "l2"'
        self.set_dist_schedule(dist_schedule)
        self.binary = binary
        self.confidence_threshold = confidence_threshold
        self.steps_dir = steps_dir
        self.current_image = ""  # for intermediate images

    def save_intermediate_img(self, x, n_iter, y_pred):
        """
        Saves intermediate images for the attack
        """
        # temp disabled
        # if len(x.shape) == 3:
        #     img_idx_path = os.path.join(self.steps_dir, self.current_image)

        #     y_pred_formatted = f"{y_pred.item():.3e}"
        #     img_path = os.path.join(
        #         img_idx_path, f"niter={n_iter:04d}_y={y_pred_formatted}.png"
        #     )

        #     save_img_threaded(x, img_path)
        # elif len(x.shape) == 4:
        #     for i in range(x.size(0)):
        #         img = x[i]
        #         img_idx_path = os.path.join(self.steps_dir, self.current_image[i])

        #         y_pred_formatted = f"{y_pred[i].item():.3e}"
        #         img_path = os.path.join(
        #             img_idx_path, f"niter={n_iter:04d}_y={y_pred_formatted}.png"
        #         )
        #         save_img_threaded(img, img_path)

    def set_dist_schedule(self, schedule):
        """
        Sets the distance schedule for the sampling looping
        """
        looper = range(self.nb_iter)
        if schedule == "none":
            schedule = [1 for _ in looper]
        elif schedule == "step":
            schedule = [0 if i <= self.nb_iter // 2 else 1 for i in looper]
        elif schedule == "linear":
            schedule = [(i + 1) / self.nb_iter for i in looper]
        self.dist_schedule = schedule

    @torch.enable_grad()
    def extract_dist_grads(self, i, x, x_adv):
        """
        Distance gradients extraction
        :param i: current step
        :param x: clean input
        :param x_adv: adversarial input
        """
        if self.dist_fn is not None:
            x_adv.requires_grad = True
            grad = torch.autograd.grad(self.dist_fn(x, x_adv), x_adv)[0]
            return self.dist_schedule[i] * grad
        return 0

    def l2_norm_proj(self, x, x_adv):
        """
        Projection over the l2 norm ball over x with a budjet of eps.
        Produce clamping at the end
        :param x: clean instance
        :param x_adv: adversarial instance
        """
        v = x_adv - x
        norms = torch.norm(v.view(x.size(0), -1), p=2, dim=1)
        norms = norms.view(-1, 1, 1, 1)
        passed = norms > self.eps
        return ((self.eps * v * passed / norms + v * (1 - passed)) + x).clamp(0, 1)

    def linf_norm_proj(self, x, x_adv):
        """
        Projection over the linf norm ball over x with a budjet of eps.
        Produce clamping at the end
        :param x: clean instance
        :param x_adv: adversarial instance
        """
        x_adv = torch.min(x + self.eps, x_adv)
        x_adv = torch.max(x - self.eps, x_adv)
        return x_adv.clamp(0, 1)

    def perturb(self, x, y=None):
        """
        Attack x in a targeted (y!=None) or untargeted way (y==None)
        :param x: input to be attacked
        :param y: optional target
        """
        self.targeted = y is not None
        self.sign = 1 if self.targeted else -1

        if not self.targeted:
            raise NotImplementedError(
                "Untargeted attack for regression not implemented yet."
            )
            # with torch.no_grad():
            #     y = self.predict(x).argmax(dim=1)

        return self.attack(x, y)

    @torch.enable_grad()
    def extract_grads(self, x, y):
        """
        Extract gradients of x w.r.t. the loss function operated on y.
        When y was none on perturb, y=f(clean x)
        """

        x.requires_grad = True
        out = self.predict(x)
        l = self.loss_fn(out, y)  # TODO: Untargeted or not
        grad = torch.autograd.grad(l, x)[0]

        return grad, out

    def attack(self, x, y):
        raise NotImplementedError("Attack not implemented.")


class ClassifierDiffusionCheckpointGradients(Attack):
    """
    Class to extract gradients from a DDPM + classifier
    combined model using the checkpoint method. Replaces
    the extract_grads function for the resource-efficient
    method.

    This method is SLOW but saves a lot of computational resources.
    From my experiments, it is faster to have a more backward steps
    than a larger batch size.
    """

    def __init__(
        self,
        predict,
        diffusion,
        ddpm,
        loss_fn,
        dist_fn,
        eps,
        nb_iter,
        norm="linf",
        step=1 / 255,
        steps=60,
        stochastic=True,
        backward_steps=1,
        dist_schedule="none",
        binary=False,
    ):
        """
        :param steps: forward/backward diffusion steps
        :param stochastic: Change the noise at each step when computing the gradients
        """

        super().__init__(
            predict=predict,
            loss_fn=loss_fn,
            dist_fn=dist_fn,
            eps=eps,
            step=step,
            nb_iter=nb_iter,
            norm=norm,
            dist_schedule=dist_schedule,
            binary=binary,
        )

        # diffusion model objects
        self.ddpm = ddpm
        self.steps = steps
        self.diffusion = diffusion
        self.stochastic = stochastic
        self.backward_steps = backward_steps
        self.noise_fn = torch.randn if stochastic else torch.zeros

    @torch.enable_grad()
    def extract_grads(self, x, y):

        timesteps = list(range(self.steps))[::-1] + [
            "c"
        ]  # the 'c' is for the classification step
        chunked_timesteps = [
            timesteps[::-1][i : i + self.backward_steps][::-1]
            for i in range(0, len(timesteps), self.backward_steps)
        ][::-1]

        B, C, H, W = x.shape
        # Precompute all noise steps.
        # To save up memory a little bit of memory, we use the same noise at each step for the same batch size.
        noises = self.noise_fn(self.steps + 1, C, H, W)

        x_orig = x.clone().detach()
        pointer = -1
        schedule = sum(chunked_timesteps[:pointer], [])
        grad = None  # for first the backward phase

        while True:

            # no grad steps
            torch.set_grad_enabled(False)
            idx = -1

            for idx, t in enumerate(schedule):
                t = torch.tensor([t] * x_orig.size(0), device=x_orig.device)

                if idx == 0:
                    x = (x - 0.5) / 0.5
                    noise = (
                        noises[0, ...]
                        .unsqueeze(dim=0)
                        .expand(x.size(0), -1, -1, -1)
                        .to(x.device)
                    )
                    x = self.diffusion.q_sample(x, t, noise=noise)
                    del noise

                x = self.forward(
                    x,
                    t,
                    idx,
                    noises[idx + 1]
                    .unsqueeze(dim=0)
                    .expand(x.size(0), -1, -1, -1)
                    .to(x.device),
                )

            diff_steps = chunked_timesteps[pointer]

            # gradient steps
            torch.set_grad_enabled(True)

            # set requires grad to true for the first iteration
            x_in = (x_orig if (idx + 1) == 0 else x).detach().requires_grad_(True)
            output = x_in

            for jdx, t in enumerate(diff_steps, start=idx + 1):

                if t == "c":  # classification step, always final step
                    output = output * 0.5 + 0.5
                    output = self.loss_fn(self.predict(output), y)

                else:  # diffusion steps
                    t = torch.tensor([t] * x_orig.size(0), device=x_orig.device)
                    if jdx == 0:
                        output = (output - 0.5) / 0.5
                        noise = (
                            noises[0, ...]
                            .unsqueeze(dim=0)
                            .expand(x.size(0), -1, -1, -1)
                            .to(x.device)
                        )
                        output = self.diffusion.q_sample(output, t, noise=noise)

                    output = self.forward(
                        output,
                        t,
                        jdx,
                        noises[jdx + 1]
                        .unsqueeze(dim=0)
                        .expand(x.size(0), -1, -1, -1)
                        .to(x.device),
                    )
            # computes gradient
            grad = torch.autograd.grad(output, x_in, grad_outputs=grad)[0]

            # breaks if schedule is empty
            if len(schedule) == 0:
                break

            pointer -= 1
            schedule = sum(chunked_timesteps[:pointer], [])

        return grad

    def forward(self, x, t, idx, noise):
        out = self.diffusion.p_mean_variance(self.ddpm, x, t, clip_denoised=True)

        x = out["mean"]

        if idx != (self.steps - 1):
            x += torch.exp(0.5 * out["log_variance"]) * noise

        return x


class ClassifierDiffusionShortcut(ClassifierDiffusionCheckpointGradients):

    def _extract_into_tensor(self, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        res = (
            torch.from_numpy(self.diffusion.sqrt_alphas_cumprod)
            .to(device=timesteps.device)[timesteps]
            .float()
        )
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def extract_grads(self, x, y):

        # DDPM unconditional forward
        with torch.no_grad():
            timesteps = list(range(self.steps))[::-1]
            x = (x - 0.5) / 0.5

            for idx, t in enumerate(timesteps):

                t = torch.tensor([t] * x.size(0), device=x.device)

                if idx == 0:
                    noise = (
                        torch.randn_like(x) if self.stochastic else torch.zeros_like(x)
                    )
                    x = self.diffusion.q_sample(x, t, noise=noise)

                if self.fix_noise:
                    noise = self.noise[idx + 1, ...].unsqueeze(dim=0)
                elif self.stochastic:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = self.forward(x, t, idx, noise)

            x_in = (x * 0.5) + 0.5
            t = torch.tensor([self.steps - 1] * x.size(0), device=x.device)

        with torch.enable_grad():
            # gradient steps shortcut
            alpha_t = self._extract_into_tensor(t, x_in.shape)
            x_in = x_in.detach().requires_grad_(True)
            loss = self.loss_fn(self.predict(x_in), y)
            grad = torch.autograd.grad(loss, x_in)[0]
            grad = grad / alpha_t

        return grad


# =======================================================
# Individual Attacks
# =======================================================


def get_attack(attack, use_checkpoint, use_shortcut=False):

    BaseAttack = Attack
    post_text = ""
    if use_checkpoint and not use_shortcut:
        post_text = " with checkpoint method"
        BaseAttack = ClassifierDiffusionCheckpointGradients
    elif not use_checkpoint and use_shortcut:
        post_text = " with shortcut method"
        BaseAttack = ClassifierDiffusionShortcut

    class NoAttack(BaseAttack):
        """
        Implement no attack.
        """

        @staticmethod
        def perturb(x, y=None):
            """
            Returns the input instance
            """
            return x

    class PGD(BaseAttack):
        """
        PGD attack
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            if self.loss_fn == "mse":
                self.loss_fn = torch.nn.MSELoss()  # Default Regression loss
            if (self.loss_fn is None) and (not self.binary):
                self.loss_fn = torch.nn.CrossEntropyLoss()
            elif (self.loss_fn is None) and self.binary:
                self.loss_fn = torch.nn.BCEWithLogitsLoss()

        @torch.no_grad()
        def attack(self, xs: torch.Tensor, ys: torch.Tensor):
            """
            Main PGD algorithm, running on batches
            """

            xs_adv = xs.clone().detach()
            projection_fn = (
                self.linf_norm_proj if self.norm == "linf" else self.l2_norm_proj
            )

            attacking_mask = torch.ones(xs.size(0)).to(xs_adv).view(-1, 1, 1, 1)
            confidence_thresholds: torch.Tensor = torch.Tensor(
                [self.confidence_threshold]
            ).to(xs)
            steps_needed = torch.zeros(xs.size(0), dtype=torch.int16)

            def track_successful_attacks(
                i: int,
                attacking_mask: torch.Tensor,
                confidence_thresholds: torch.Tensor,
                confidences: torch.Tensor,
                steps_needed: torch.Tensor,
            ):
                successful_attacks = confidences <= confidence_thresholds
                attacking_mask[successful_attacks] = 0
                steps_needed = torch.where(
                    attacking_mask.view(-1).cpu() == 1,
                    i,
                    steps_needed,
                )
                return attacking_mask, steps_needed

            with tqdm(range(self.nb_iter), desc="PGD") as pbar:

                def update_pbar(attacking_mask, y_hat, confidences):
                    pbar.set_postfix(
                        confidence_mean=confidences.mean().item(),
                        regr=[f"{val:.4f}" for val in y_hat.cpu().view(-1).tolist()],
                        regr_attacking=attacking_mask.view(-1).int().cpu().tolist(),
                        max_gpu_GB=torch.cuda.max_memory_reserved() / 1e9,
                    )

                for i in pbar:
                    grads, y_hat = self.extract_grads(xs_adv, ys)
                    grads = self.sign * grads + self.extract_dist_grads(
                        i, xs, xs_adv.clone().detach()
                    )
                    xs_adv -= grads.sign() * self.step * attacking_mask
                    xs_adv = projection_fn(xs, xs_adv)

                    # y_hat = self.predictor(xs_adv)
                    confidences = get_regr_confidence(ys, y_hat)

                    self.save_intermediate_img(xs_adv, i, y_hat)
                    update_pbar(attacking_mask, y_hat, confidences)

                    # Handle the images that have reached the confidence threshold and update mask
                    attacking_mask, steps_needed = track_successful_attacks(
                        i,
                        attacking_mask,
                        confidence_thresholds,
                        confidences,
                        steps_needed,
                    )

                    if attacking_mask.sum() == 0:
                        break

            successes: list[bool] = (~attacking_mask.bool().view(-1)).cpu().tolist()
            return xs_adv, successes, steps_needed.tolist()

    class GradientDescent(BaseAttack):
        """
        GD attack. Same as PGD but without the sign function
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            if (self.loss_fn is None) and (not self.binary):
                self.loss_fn = torch.nn.CrossEntropyLoss()
            elif (self.loss_fn is None) and self.binary:
                self.loss_fn = torch.nn.BCEWithLogitsLoss()

        @torch.no_grad()
        def attack(self, x, y):
            """
            Main GD algorithm
            """

            x_adv = x.clone().detach()
            projection_fn = (
                self.linf_norm_proj if self.norm == "linf" else self.l2_norm_proj
            )

            for i in range(self.nb_iter):
                grad = self.sign * self.extract_grads(
                    x_adv, y
                ) + self.extract_dist_grads(i, x, x_adv.clone().detach())
                x_adv -= grad * self.step
                x_adv = projection_fn(x, x_adv)

            return x_adv

    class CW(BaseAttack):
        """
        C&W attack.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            if (self.loss_fn is None) and (not self.binary):
                self.loss_fn = MultiClassCW()
            elif (self.loss_fn is None) and self.binary:
                self.loss_fn = BinaryCW()
            # in the C&W method, they use a constant to avoid 0 or inf gradients
            self._c = 1e-5

        @torch.no_grad()
        def attack(self, x, y):
            """
            Main C&W algorithm
            """
            assert self.targeted, "C&W is a targeted attack"

            x_adv = x.clone().detach()
            projection_fn = (
                self.linf_norm_proj if self.norm == "linf" else self.l2_norm_proj
            )

            # instantiate w_i
            # w = torch.zeros_like(x)

            # instantiate w_i as the inverse of x such that (1 / 2) * (torch.tanh(w) + 1) = x
            w = torch.atanh(2 * x.clone().detach() / (1 + self._c) - 1)

            for i in range(self.nb_iter):
                # these are the gradients wrt (1 / 2) * (torch.tanh(w) + 1)
                x_adv = (torch.tanh(w) + 1) * (self._c + 1 / 2)
                grad = self.sign * self.extract_grads(
                    x_adv, y
                ) + self.extract_dist_grads(i, x, x_adv.clone().detach())

                # manually optimize w via chain rule
                w -= self.step * grad * (self._c + 1 / 2) * (1 - torch.tanh(w).pow(2))
                w = w.clamp(-3, 3)  # to avoid 0/inf grads

            return x_adv

    print(f"Loading {attack}" + post_text)

    if attack == "None":
        return NoAttack
    elif attack == "PGD":
        return PGD
    elif attack == "GD":
        return GradientDescent
    elif attack == "CW":
        print("** Warning. C&W attack has no epsilon bound (except for [0, 1])!! **")
        return CW
    elif attack == "Adam":
        return AdamAttack
    else:
        raise NotImplementedError(f"Attack {attack} is not implemented.")


# =======================================================
# Additional Losses and Function
# =======================================================


class BinaryCW(torch.nn.Module):

    relu = torch.nn.ReLU(inplace=True)

    def forward(self, logits, target):
        sign = torch.ones_like(target)
        sign[target == 0] = -1
        F_t = logits * sign
        return self.relu(-2 * F_t).sum()


class MultiClassCW(torch.nn.Module):
    relu = torch.nn.ReLU(inplace=True)

    def forward(self, logits, target):
        F_t = logits[list(range(len(target))), target]
        wo_t = logits
        # replace the target for -inf to take the max
        wo_t[list(range(len(target))), target] = -float("inf")
        F_c = wo_t.max(dim=1)[0]
        return self.relu(F_c - F_t).sum()


# =======================================================
# Custom Attacks
# =======================================================
class AdamAttack(Attack):
    """
    Adam based attack
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.loss_fn == "mse":
            self.loss_fn = torch.nn.MSELoss()  # Default Regression loss
        if (self.loss_fn is None) and (not self.binary):
            self.loss_fn = torch.nn.CrossEntropyLoss()
        elif (self.loss_fn is None) and self.binary:
            self.loss_fn = torch.nn.BCEWithLogitsLoss()

        if self.dist_fn is None:
            self.dist_fn = lambda x, y: 0

    @torch.no_grad()
    def attack(self, xs, ys):
        """
        Main attack algorithm

        :param x: The input tensor.
        :param y: The target labels tensor.
        :param img_idxs: The indices of the images (used to save intermediate attack images).
        """

        xs_adv_list = xs.clone().detach().split(1, dim=0)
        for x in xs_adv_list:
            x.requires_grad_()

        confidence_thresholds: torch.Tensor = torch.Tensor(
            [self.confidence_threshold]
        ).to(xs)
        steps_needed = torch.zeros(xs.size(0), dtype=torch.int16)

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
                    xs_adv_list[i].requires_grad = False

            done = all([not x.requires_grad for x in xs_adv_list])
            return steps_needed, done

        optimizer = torch.optim.Adam(xs_adv_list, lr=self.step)
        print(f"Init Adam with step {self.step}")
        with tqdm(range(self.nb_iter), desc="Adam") as pbar:

            def update_pbar(y_hat, confidences):
                pbar.set_postfix(
                    confidence_mean=confidences.mean().item(),
                    regr=[f"{val:.4f}" for val in y_hat.cpu().view(-1).tolist()],
                    regr_attacking=[1 if x.requires_grad else 0 for x in xs_adv_list],
                    max_gpu_GB=torch.cuda.max_memory_reserved() / 1e9,
                    steps_needed=steps_needed.tolist(),
                )

            with torch.enable_grad():
                for i in pbar:
                    # Filter Function -> Classifier loss + Distance function
                    xs_adv = torch.cat(xs_adv_list, dim=0)
                    y_hat = self.predict(xs_adv)
                    loss = self.loss_fn(y_hat, ys)
                    dist_x = self.dist_fn(xs, xs_adv)
                    total_loss = loss + dist_x

                    confidences = get_regr_confidence(ys, y_hat)
                    self.save_intermediate_img(xs_adv, i, y_hat)

                    # Handle the images that have reached the confidence threshold and update mask
                    steps_needed, done = track_successful_attacks(
                        i,
                        confidence_thresholds,
                        confidences,
                        steps_needed,
                    )
                    update_pbar(y_hat, confidences)
                    if done:
                        break

                    total_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

        successes: list[bool] = [not x.requires_grad for x in xs_adv_list]
        return xs_adv, successes, steps_needed.tolist()
