from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass
import torch

from .guidance_utils import MomentumBuffer, constant_guidance, linear_guidance, exponential_guidance, adaptative_projected_guidance, zero_star_guidance
from .error import check_existing_guidance_method, check_guidance_parameters

class GuidanceMethod(ABC):
    """
    Base class for all guidance methods.
    Note: Abstract class.
    """

    def __init__(self):
        pass
    
    def reset(self):
        pass

    @abstractmethod
    def predict_velocity_field(self, ctx: GuidanceContext) -> torch.Tensor:
        """
        Compute the guided model prediction for one denoising step.
        """
        raise NotImplementedError


class ConstantGuidanceMethod(GuidanceMethod):

    def predict_velocity_field(self, ctx: GuidanceContext) -> torch.Tensor:
        pred_uncond, pred_cond = ctx.pipeline._predict_model(latents=ctx.latents, t=ctx.t, 
                                                             prompt_embeds=ctx.prompt_embeds, 
                                                             pooled_prompt_embeds=ctx.pooled_prompt_embeds, 
                                                             do_cfg=True)
        
        return constant_guidance(pred_uncond, pred_cond, ctx.guidance_scale)


class LinearGuidanceMethod(GuidanceMethod):

    def predict_velocity_field(self, ctx: GuidanceContext) -> torch.Tensor:
        pred_uncond, pred_cond = ctx.pipeline._predict_model(latents=ctx.latents, t=ctx.t, 
                                                             prompt_embeds=ctx.prompt_embeds,
                                                             pooled_prompt_embeds=ctx.pooled_prompt_embeds,
                                                             do_cfg=True)
        
        return linear_guidance(pred_uncond, pred_cond, ctx.guidance_scale, ctx.normalized_time())


class ExponentialGuidanceMethod(GuidanceMethod):

    def predict_velocity_field(self, ctx: GuidanceContext) -> torch.Tensor:
        pred_uncond, pred_cond = ctx.pipeline._predict_model(latents=ctx.latents, t=ctx.t, 
                                                             prompt_embeds=ctx.prompt_embeds, 
                                                             pooled_prompt_embeds=ctx.pooled_prompt_embeds, 
                                                             do_cfg=True)
        
        return exponential_guidance(pred_uncond, pred_cond, ctx.guidance_scale, ctx.normalized_time())


class APGGuidanceMethod(GuidanceMethod):

    def __init__(self, momentum_value: float = 0.9, eta: float = 1.0, norm_threshold: float = 0.0):
        super().__init__()
        self.momentum_value = momentum_value
        self.momentum_buffer = MomentumBuffer(momentum_value)
        self.eta = eta
        self.norm_threshold = norm_threshold
        
    def reset(self):
        self.momentum_buffer = MomentumBuffer(self.momentum_value)

    def predict_velocity_field(self, ctx: GuidanceContext) -> torch.Tensor:
        pred_uncond, pred_cond = ctx.pipeline._predict_model(latents=ctx.latents, t=ctx.t, 
                                                             prompt_embeds=ctx.prompt_embeds,
                                                             pooled_prompt_embeds=ctx.pooled_prompt_embeds, 
                                                             do_cfg=True)
        
        apg_state = {"momentum_buffer": self.momentum_buffer, "eta": self.eta, "norm_threshold": self.norm_threshold}

        return adaptative_projected_guidance(pred_uncond, pred_cond, ctx.guidance_scale, ctx.normalized_time(), ctx.latents, apg_state)


class RectifiedPPGuidanceMethod(GuidanceMethod):

    def __init__(self, alpha_scale: Optional[float] = None):
        super().__init__()
        self.alpha_scale = alpha_scale

    def _compute_dt(self, ctx: GuidanceContext) -> torch.Tensor:
        if ctx.step_index < len(ctx.timesteps) - 1:
            return ctx.timesteps[ctx.step_index] - ctx.timesteps[ctx.step_index + 1]
        return ctx.timesteps[ctx.step_index]

    def _compute_alpha_t(self, ctx: GuidanceContext) -> torch.Tensor:
        if self.alpha_scale is not None:
            return torch.as_tensor(self.alpha_scale, device=ctx.latents.device, dtype=ctx.latents.dtype)
        return torch.as_tensor(ctx.guidance_scale, device=ctx.latents.device, dtype=ctx.latents.dtype)

    def predict_velocity_field(self, ctx: GuidanceContext) -> torch.Tensor:
        v_cond = ctx.pipeline._predict_model(latents=ctx.latents, t=ctx.t, 
                                             prompt_embeds=ctx.original_prompt_embeds, 
                                             pooled_prompt_embeds=ctx.original_pooled_prompt_embeds, 
                                             do_cfg=False)
        dt = self._compute_dt(ctx)
        x_mid = ctx.latents + 0.5 * dt * v_cond
        t_mid = ctx.t - 0.5 * dt

        v_uncond_mid, v_cond_mid = ctx.pipeline._predict_model(latents=x_mid, t=t_mid, 
                                                               prompt_embeds=ctx.prompt_embeds, 
                                                               pooled_prompt_embeds=ctx.pooled_prompt_embeds,
                                                               do_cfg=True)
        alpha_t = self._compute_alpha_t(ctx)
        return v_cond + alpha_t * (v_cond_mid - v_uncond_mid)


class ZeroStarGuidanceMethod(GuidanceMethod):

    def __init__(self, zero_steps: int = 0, use_zero_init: bool = False):
        super().__init__()
        self.zero_steps = zero_steps
        self.use_zero_init = use_zero_init

    def predict_velocity_field(self, ctx: GuidanceContext) -> torch.Tensor:
        pred_uncond, pred_cond = ctx.pipeline._predict_model(latents=ctx.latents, t=ctx.t, 
                                                             prompt_embeds=ctx.prompt_embeds,
                                                             pooled_prompt_embeds=ctx.pooled_prompt_embeds, 
                                                             do_cfg=True)
        
        return zero_star_guidance(pred_uncond, pred_cond, ctx.guidance_scale, 
                                  self.zero_steps, self.use_zero_init, ctx.step_index)


def build_guidance_method(guidance_type: str, params: Optional[dict[str, Any]] = None) -> GuidanceMethod:
    
    check_existing_guidance_method(guidance_type)
    check_guidance_parameters(guidance_type, params)

    if guidance_type == "constant":
        return ConstantGuidanceMethod()

    if guidance_type == "linear":
        return LinearGuidanceMethod()

    if guidance_type == "exponential":
        return ExponentialGuidanceMethod()

    if guidance_type == "APG":
        return APGGuidanceMethod(**params)

    if guidance_type == "rectified_pp":
        return RectifiedPPGuidanceMethod(**params)

    if guidance_type == "zero_star":
        return ZeroStarGuidanceMethod(**params)


@dataclass
class GuidanceContext:
    """
    Context object passed to a guidance method for one denoising step.

    It stores references to the current pipeline state and tensors needed
    to predict the guided velocity field / noise prediction.
    """

    # Pipeline reference (used to access helper methods such as _predict_model)
    pipeline: Any

    # Current denoising state
    latents: torch.Tensor
    t: torch.Tensor
    timestep: torch.Tensor
    step_index: int
    timesteps: torch.Tensor

    # Embeddings used for CFG forward
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor

    # Conditional-only embeddings (before concatenation with negative prompt)
    original_prompt_embeds: Optional[torch.Tensor] = None
    original_pooled_prompt_embeds: Optional[torch.Tensor] = None

    # Global parameters
    guidance_scale: float = 1.0
    joint_attention_kwargs: Optional[dict[str, Any]] = None
    do_classifier_free_guidance: bool = True

    def normalized_time(self) -> torch.Tensor:
        """
        Return normalized time t / t0.
        """
        return self.t / self.timesteps[0]