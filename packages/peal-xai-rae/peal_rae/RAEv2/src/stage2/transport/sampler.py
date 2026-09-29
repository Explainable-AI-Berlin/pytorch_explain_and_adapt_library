import torch as th


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


class Sampler:
    def __init__(self, transport, guidance_config):
        self.transport = transport
        self.drift = self.transport.get_drift()
        self.guidance_config = guidance_config
        self.omega = guidance_config.cfg.scale
        self.t_start = guidance_config.cfg.t_min
        self.t_end = guidance_config.cfg.t_max

    def sample_ode(self, *, num_steps=50):
        t_grid = th.linspace(1.0, 0.0, num_steps + 1)
        shift = self.transport.time_dist_shift
        t_grid = shift * t_grid / (1 + (shift - 1) * t_grid)

        def sample_fn(x, model, **model_kwargs):
            device = _first_tensor(x).device
            t_steps = t_grid.to(device)
            B = _batch_size(x)

            model_kwargs_ = model_kwargs.copy()
            for k, v in (("omega", self.omega), ("t_start", self.t_start), ("t_end", self.t_end)):
                if v is not None:
                    model_kwargs_[k] = th.full((B,), v, device=device)

            trajectory = [x]
            for i in range(num_steps):
                h = t_steps[i] - t_steps[i + 1]
                t_batch = th.full((B,), t_steps[i].item(), device=device)
                d_cur = self.drift(x, t_batch, model, **model_kwargs_)
                x = _state_map(lambda xi, di: xi - h * di, x, d_cur)
                trajectory.append(x)

            if isinstance(x, tuple):
                return tuple(th.stack([step[j] for step in trajectory], dim=0) for j in range(len(x)))
            return th.stack(trajectory, dim=0)

        return sample_fn
