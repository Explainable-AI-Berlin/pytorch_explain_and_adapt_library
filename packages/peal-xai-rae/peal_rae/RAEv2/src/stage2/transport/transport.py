import torch as th
import torch.nn.functional as F

from stage2.utils import apply_cfg_dropout


def _is_state_tuple(x):
    return isinstance(x, tuple)


def _first_tensor(x):
    return x[0] if isinstance(x, tuple) else x


def _batch_size(x):
    return _first_tensor(x).shape[0]


def _state_map(fn, *states):
    ref = states[0]
    if th.is_tensor(ref):
        return fn(*states)
    if isinstance(ref, tuple):
        return tuple(_state_map(fn, *parts) for parts in zip(*states))
    raise TypeError(f"Unsupported state type: {type(ref)}")


def _expand_t(t, x):
    return t.view(t.size(0), *([1] * (len(x.size()) - 1)))


def _randn_like_state(x):
    return _state_map(lambda a: th.randn_like(a), x)


def _lerp_state(a, b, t):
    return _state_map(lambda ai, bi: (1 - _expand_t(t, ai)) * ai + _expand_t(t, ai) * bi, a, b)


def _velocity_target(xt, x1, t, t_eps):
    return _state_map(lambda xti, x1i: (xti - x1i) / _expand_t(t, xti).clamp_min(t_eps), xt, x1)


def _mse_terms(output, target):
    if th.is_tensor(output):
        return (output - target) ** 2
    return tuple(_mse_terms(oi, ti) for oi, ti in zip(output, target))


def _sum_terms(x):
    if th.is_tensor(x):
        return x
    out = None
    for term in x:
        out = term if out is None else out + term
    return out


def get_time_sampler(time_dist_type: str):
    parts = time_dist_type.split("_")
    name = parts[0]
    if name == "logit-normal":
        assert len(parts) == 3, f"Expected 'logit-normal_MU_SIGMA', got '{time_dist_type}'"
        mu, sigma = float(parts[1]), float(parts[2])
        assert sigma > 0, "sigma must be > 0"
        return lambda bs: (th.randn(bs) * sigma + mu).sigmoid()
    raise NotImplementedError(f"Unknown time distribution: {time_dist_type}")


class Transport:
    def __init__(self, prediction="velocity", time_dist_type="logit-normal_0_1", time_dist_shift=1.0, t_eps=0.05):
        self.prediction = prediction
        self.time_dist_type = time_dist_type
        self.time_dist_shift = time_dist_shift
        self.t_eps = t_eps
        self.time_sampler = get_time_sampler(time_dist_type)

    def sample(self, x1):
        x0 = _randn_like_state(x1)
        t = self.time_sampler(_batch_size(x1)).to(_first_tensor(x1))
        t = self.time_dist_shift * t / (1 + (self.time_dist_shift - 1) * t)
        return t, x0, x1

    def training_losses(
        self,
        model,
        x1,
        model_kwargs={},
        model_kwargs_null={},
        z_clean=None,
        repa_coeff=None,
        base_model_coeff=1.0,
        cfg_dropout_prob=0.1,
        aux_loss_weight=1.0,
    ):
        model_kwargs, _ = apply_cfg_dropout(model_kwargs, model_kwargs_null, cfg_dropout_prob)

        t, x0, x1 = self.sample(x1)
        xt = _lerp_state(x1, x0, t)
        vt = _velocity_target(xt, x1, t, self.t_eps)

        enable_repa = z_clean is not None and repa_coeff is not None
        if enable_repa:
            model_output, zt_pred = model(xt, t, return_intermediate=True, **model_kwargs)
        else:
            model_output = model(xt, t, **model_kwargs)
            zt_pred = None

        base_output = None
        if not _is_state_tuple(x1) and isinstance(model_output, tuple) and len(model_output) == 2:
            model_output, base_output = model_output

        model_pred = self.convert_model_pred(model_output, xt, t)
        if _is_state_tuple(x1):
            if not _is_state_tuple(model_pred) or len(model_pred) != 2:
                raise ValueError("Joint Stage-2 state requires the model prediction to be a (patch, aux) tuple.")
            patch_pred, aux_pred = model_pred
            patch_target, aux_target = vt
            loss_patch = F.mse_loss(patch_pred, patch_target)
            loss_aux = F.mse_loss(aux_pred, aux_target)
            terms = {
                "loss_patch": loss_patch,
                "loss_aux": loss_aux,
                "loss": loss_patch + float(aux_loss_weight) * loss_aux,
            }
        else:
            loss_terms = _sum_terms(_mse_terms(model_pred, vt))
            terms = {"loss": loss_terms}
        if base_output is not None:
            loss_base = _sum_terms(_mse_terms(self.convert_model_pred(base_output, xt, t), vt))
            terms["loss"] = terms["loss"] + base_model_coeff * loss_base
            terms["loss_base"] = loss_base
        if enable_repa and zt_pred is not None:
            terms["loss_repa"] = repa_coeff * F.mse_loss(zt_pred, z_clean)
        return terms

    def convert_model_pred(self, output, xt, t):
        if self.prediction == "velocity":
            return output
        if self.prediction == "x":
            return _state_map(
                lambda outi, xti: (xti - outi) / _expand_t(t, xti).clamp_min(self.t_eps),
                output,
                xt,
            )
        raise NotImplementedError(f"Unsupported prediction mode: {self.prediction}")

    def get_drift(self):
        def body_fn(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            if not _is_state_tuple(x) and isinstance(model_output, tuple):
                model_output = model_output[0]
            return self.convert_model_pred(model_output, x, t)

        return body_fn
