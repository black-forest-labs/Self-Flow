"""
Self-Flow Sampling Utilities.

This module contains the sampling logic for Self-Flow diffusion models,
including the SDE integrators and transport path definitions.
"""

import enum
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import torch


#################################################################################
#                              Configuration                                    #
#################################################################################

Choice_PathType = Literal["Linear", "GVP", "VP"]
Choice_Prediction = Literal["velocity", "score", "noise"]
Choice_LossWeight = Optional[Literal["velocity", "likelihood"]]
Choice_SamplingODE = Literal["heun2"]
Choice_SamplingSDE = Literal["Euler", "Heun"]
Choice_Diffusion = Literal[
    "constant", "SBDM", "sigma",
    "linear", "decreasing", "increasing-decreasing"
]
Choice_LastStep = Optional[Literal["Mean", "Tweedie", "Euler"]]


@dataclass
class TransportConfig:
    path_type: Choice_PathType = "Linear"
    prediction: Choice_Prediction = "velocity"
    loss_weight: Choice_LossWeight = None
    sample_eps: Optional[float] = None
    train_eps: Optional[float] = None


@dataclass
class ODEConfig:
    sampling_method: Choice_SamplingODE = "heun2"
    atol: float = 1e-6
    rtol: float = 1e-3
    reverse: bool = False
    likelihood: bool = False


@dataclass
class SDEConfig:
    sampling_method: Choice_SamplingSDE = "Euler"
    diffusion_form: Choice_Diffusion = "sigma"
    diffusion_norm: float = 1.0
    last_step: Choice_LastStep = "Mean"
    last_step_size: float = 0.04


@dataclass
class Config:
    transport: TransportConfig = field(default_factory=TransportConfig)
    ode: ODEConfig = field(default_factory=ODEConfig)
    sde: SDEConfig = field(default_factory=SDEConfig)
    num_steps: int = 64
    cfg_scale: float = 1


#################################################################################
#                              Path Functions                                   #
#################################################################################

def expand_t_like_x(t, x):
    """Function to reshape time t to broadcastable dimension of x."""
    dims = [1] * (len(x.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t


class ICPlan:
    """Linear Coupling Plan (Interpolant Conditional Plan)."""

    def __init__(self, sigma=0.0):
        self.sigma = sigma

    def compute_alpha_t(self, t):
        """Compute the data coefficient along the path."""
        return t, 1

    def compute_sigma_t(self, t):
        """Compute the noise coefficient along the path."""
        return 1 - t, -1

    def compute_d_alpha_alpha_ratio_t(self, t):
        """Compute the ratio between d_alpha and alpha."""
        return 1 / t

    def compute_drift(self, x, t):
        """Compute the drift term for SDE."""
        t = expand_t_like_x(t, x)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        drift = alpha_ratio * x
        diffusion = alpha_ratio * (sigma_t ** 2) - sigma_t * d_sigma_t
        return -drift, diffusion

    def compute_diffusion(self, x, t, form="constant", norm=1.0):
        """Compute the diffusion term of the SDE."""
        t = expand_t_like_x(t, x)
        choices = {
            "constant": norm,
            "SBDM": norm * self.compute_drift(x, t)[1],
            "sigma": norm * self.compute_sigma_t(t)[0],
            "linear": norm * (1 - t),
            "decreasing": 0.25 * (norm * torch.cos(np.pi * t) + 1) ** 2,
            "increasing-decreasing": norm * torch.sin(np.pi * t) ** 2,
        }
        try:
            diffusion = choices[form]
        except KeyError:
            raise NotImplementedError(f"Diffusion form {form} not implemented")
        return diffusion

    def get_score_from_velocity(self, velocity, x, t):
        """Transform velocity prediction model to score."""
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * velocity - mean) / var
        return score


#################################################################################
#                              Model Type Enums                                 #
#################################################################################

class ModelType(enum.Enum):
    NOISE = enum.auto()
    SCORE = enum.auto()
    VELOCITY = enum.auto()


class PathType(enum.Enum):
    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()


class WeightType(enum.Enum):
    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


#################################################################################
#                              Transport                                        #
#################################################################################

class Transport:
    """Transport class for flow matching."""

    def __init__(
        self,
        *,
        model_type,
        path_type,
        loss_type,
        train_eps,
        sample_eps,
    ):
        path_options = {
            PathType.LINEAR: ICPlan,
        }
        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps

    def check_interval(
        self,
        train_eps,
        sample_eps,
        *,
        diffusion_form="SBDM",
        sde=False,
        reverse=False,
        eval=False,
        last_step_size=0.0,
    ):
        t0 = 0
        t1 = 1
        eps = train_eps if not eval else sample_eps

        if self.model_type != ModelType.VELOCITY or sde:
            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1

    def get_drift_from_model_output(self):
        """Get the drift function for velocity model."""
        def velocity_ode(x, t, model_output):
            return model_output
        return velocity_ode

    def get_score_from_model_output(self):
        """Get the score function from model output."""
        return lambda x, t, model_output: self.path_sampler.get_score_from_velocity(model_output, x, t)


def create_transport(
    path_type='Linear',
    prediction="velocity",
    loss_weight=None,
    train_eps=None,
    sample_eps=None,
):
    """Create a Transport object."""
    model_type = ModelType.VELOCITY
    loss_type = WeightType.NONE
    path_choice = {"Linear": PathType.LINEAR}
    path_type = path_choice[path_type]

    # For velocity & LINEAR, eps = 0
    train_eps = 0
    sample_eps = 0

    return Transport(
        model_type=model_type,
        path_type=path_type,
        loss_type=loss_type,
        train_eps=train_eps,
        sample_eps=sample_eps,
    )


#################################################################################
#                              SDE Integrator                                   #
#################################################################################

class sde:
    """SDE solver class."""

    def __init__(
        self,
        drift,
        diffusion,
        *,
        t0,
        t1,
        num_steps,
        sampler_type,
    ):
        assert t0 < t1, "SDE sampler has to be in forward time"
        self.num_timesteps = num_steps
        self.t = torch.linspace(t0, t1, num_steps)
        self.dt = self.t[1] - self.t[0]
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type

    def __Euler_Maruyama_step(self, x, mean_x, t, model, **model_kwargs):
        w_cur = torch.randn(x.size()).to(x)
        t = torch.ones(x.size(0)).to(x) * t
        dw = w_cur * torch.sqrt(self.dt)
        drift = self.drift(x, t, model, **model_kwargs)
        diffusion = self.diffusion(x, t)
        mean_x = x + drift * self.dt
        x = mean_x + torch.sqrt(2 * diffusion) * dw
        return x, mean_x

    def __Heun_step(self, x, _, t, model, **model_kwargs):
        w_cur = torch.randn(x.size()).to(x)
        dw = w_cur * torch.sqrt(self.dt)
        t_cur = torch.ones(x.size(0)).to(x) * t
        diffusion = self.diffusion(x, t_cur)
        xhat = x + torch.sqrt(2 * diffusion) * dw
        K1 = self.drift(xhat, t_cur, model, **model_kwargs)
        xp = xhat + self.dt * K1
        K2 = self.drift(xp, t_cur + self.dt, model, **model_kwargs)
        return xhat + 0.5 * self.dt * (K1 + K2), xhat

    def __forward_fn(self):
        sampler_dict = {
            "Euler": self.__Euler_Maruyama_step,
            "Heun": self.__Heun_step,
        }
        try:
            sampler = sampler_dict[self.sampler_type]
        except:
            raise NotImplementedError("Sampler type not implemented.")
        return sampler

    def sample(self, init, model, **model_kwargs):
        """Forward loop of SDE."""
        x = init
        mean_x = init
        samples = []
        sampler = self.__forward_fn()
        for ti in self.t[:-1]:
            with torch.no_grad():
                x, mean_x = sampler(x, mean_x, ti, model, **model_kwargs)
                samples.append(x)
        return samples


#################################################################################
#                              Fixed Sampler                                    #
#################################################################################

class FixedSampler:
    """Fixed SDE Sampler."""

    def __init__(self, transport):
        self.transport = transport
        self.drift = self.transport.get_drift_from_model_output()
        self.score = self.transport.get_score_from_model_output()

    def __get_sde_diffusion_and_drift(
        self,
        *,
        diffusion_form="SBDM",
        diffusion_norm=1.0,
    ):
        def diffusion_fn(x, t):
            diffusion = self.transport.path_sampler.compute_diffusion(
                x, t, form=diffusion_form, norm=diffusion_norm
            )
            return diffusion

        def sde_drift(x, t, model, **kwargs):
            model_output = model(x, t, **kwargs)
            return self.drift(x, t, model_output) + diffusion_fn(x, t) * self.score(
                x, t, model_output
            )

        return sde_drift, diffusion_fn

    def __get_last_step(
        self,
        sde_drift,
        *,
        last_step,
        last_step_size,
    ):
        """Get the last step function of the SDE solver."""
        if last_step is None:
            last_step_fn = lambda x, t, model, **model_kwargs: x
        elif last_step == "Mean":
            last_step_fn = (
                lambda x, t, model, **model_kwargs: x
                + sde_drift(x, t, model, **model_kwargs) * last_step_size
            )
        elif last_step == "Euler":
            last_step_fn = (
                lambda x, t, model, **model_kwargs: x
                + self.drift(x, t, model(x, t, **model_kwargs)) * last_step_size
            )
        else:
            raise NotImplementedError()
        return last_step_fn

    def sample_sde(
        self,
        *,
        sampling_method="Euler",
        diffusion_form="SBDM",
        diffusion_norm=1.0,
        last_step="Mean",
        last_step_size=0.04,
        num_steps=250,
    ):
        """Returns a sampling function with given SDE settings."""
        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
        )

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        _sde = sde(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method,
        )

        last_step_fn = self.__get_last_step(
            sde_drift, last_step=last_step, last_step_size=last_step_size
        )

        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, **model_kwargs)
            ts = torch.ones(init.size(0), device=init.device) * t1
            x = last_step_fn(xs[-1], ts, model, **model_kwargs)
            xs.append(x)
            assert len(xs) == num_steps, "Samples does not match the number of steps"
            return xs

        return _sample


#################################################################################
#                              Guidance                                         #
#################################################################################

def vanilla_guidance(x: torch.Tensor, cfg_val: float | torch.Tensor):
    """Apply classifier-free guidance to model output."""
    x_u, x_c = x.chunk(2)
    if isinstance(cfg_val, torch.Tensor):
        while cfg_val.dim() < x_u.dim():
            cfg_val = cfg_val.unsqueeze(-1)
    x = x_u + cfg_val * (x_c - x_u)
    return x


#################################################################################
#                              Denoise Loop                                     #
#################################################################################

def denoise_loop(
    *,
    model,
    num_steps,
    cfg_scale=None,
    guidance_low=0.0,
    guidance_high=1.0,
    mode="ODE",
    sampling_method="euler",
    reverse: bool = True,
    **model_kwargs,
):
    """
    Main denoising loop for Self-Flow sampling.
    """
    args = Config()
    args.num_steps = num_steps
    
    transport = create_transport(
        args.transport.path_type,
        args.transport.prediction,
        args.transport.loss_weight,
        args.transport.train_eps,
        args.transport.sample_eps,
    )

    if mode == "SDE":
        sampler = FixedSampler(transport)
        sample_fn = sampler.sample_sde(
            sampling_method=args.sde.sampling_method,
            diffusion_form=args.sde.diffusion_form,
            diffusion_norm=args.sde.diffusion_norm,
            last_step=args.sde.last_step,
            last_step_size=args.sde.last_step_size,
            num_steps=args.num_steps,
        )
    else:
        raise NotImplementedError("Only SDE mode is currently supported")

    def model_fn(x, t, **kwargs):
        t_orig = t
        t = 1 - t if reverse else t

        # Check if we should apply CFG based on guidance scheduling
        if cfg_scale is not None and cfg_scale > 1.0:
            if torch.is_tensor(t):
                apply_cfg = torch.all((guidance_low <= t) & (t <= guidance_high)).item()
            else:
                apply_cfg = guidance_low <= t <= guidance_high
        else:
            apply_cfg = False

        if apply_cfg and mode == "SDE":
            bs = x.shape[0]
            assert bs % 2 == 0
            x = torch.concat((x[bs // 2:], x[bs // 2:]))

        pred = model(x, timesteps=t, **kwargs).to(torch.float32)
        
        if apply_cfg:
            pred = vanilla_guidance(pred, cfg_val=cfg_scale)
            pred = torch.cat((pred, pred))

        return -pred if reverse else pred

    samples = sample_fn(model_kwargs.pop("x"), model_fn, **model_kwargs)[-1]
    return samples
