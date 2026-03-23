import numpy as np
import torch

from guidance_utils import _project, _to_denoise, _to_noise

def constant_guidance(noise_pred_uncond, noise_pred_text, guidance_scale):
    """
    Applies constant guidance to the noise prediction.
    
    Args:
        noise_pred_uncond: The noise prediction for the unconditional input.
        noise_pred_text: The noise prediction for the text input.
        guidance_scale: set so that integral from T to 0 of omega(t) = guidance_scale * T.
        time: current time step, normalized in [0, 1]; noisy = 1, denoised = 0.
    
    Returns:
        The guided noise prediction.
    """
    omega = guidance_scale
    return noise_pred_uncond + (noise_pred_text - noise_pred_uncond) * omega

def linear_guidance(noise_pred_uncond, noise_pred_text, guidance_scale, time):
    """ 
    Applies increasing linear guidance to the noise prediction.
    
    Args:
        noise_pred_uncond: The noise prediction for the unconditional input.
        noise_pred_text: The noise prediction for the text input.
        guidance_scale: set so that integral from T to 0 of omega(t) = guidance_scale * T.
        time: current time step, normalized in [0, 1]; noisy = 1, denoised = 0.
    
    Returns:
        The guided noise prediction.
    """
    # Calculate the linear scaling factor based on the current step
    omega = 2 * (1-time) * guidance_scale
    return noise_pred_uncond + (noise_pred_text - noise_pred_uncond) * omega

def exponential_guidance(noise_pred_uncond, noise_pred_text, guidance_scale, time):
    """ 
    Applies increasing exponential guidance to the noise prediction.
    
    Args:
        noise_pred_uncond: The noise prediction for the unconditional input.
        noise_pred_text: The noise prediction for the text input.
        guidance_scale: set so that integral from T to 0 of omega(t) = guidance_scale * T.
        time: current time step, normalized in [0, 1]; noisy = 1, denoised = 0.
    
    Returns:
        The guided noise prediction.
    """
    # Calculate the exponential scaling factor based on the current step
    alpha = (guidance_scale / (np.exp(1) - 1))
    omega = alpha * (np.exp(1-time))
    return noise_pred_uncond + (noise_pred_text - noise_pred_uncond) * omega


def adaptative_projected_guidance(noise_pred_uncond, noise_pred_text, guidance_scale, time, latents, APG_parameters):
    """
    Implements the Adaptative Projected Guidance (APG) method for guiding the noise prediction in a diffusion model.

    Args:
        noise_pred_uncond: The noise prediction for the unconditional input, shape (B, C, H, W).
        noise_pred_text: The noise prediction for the text input, shape (B, C, H, W).
        guidance_scale: The scale of the guidance to apply.
        time: The current time step in the diffusion process.
        APG_parameters: A dictionary containing the parameters for the APG method, including:
            - momentum_buffer: An instance of MomentumBuffer to store the momentum of the updates.
            - eta: The scaling factor for the parallel component of the update.
            - norm_threshold: The maximum allowed norm for the update.
        latents: The current noisy latents at time t, shape (B, C, H, W).
    
    Returns:
        The guided noise prediction, shape (B, C, H, W).
    """
    x0_uncond = _to_denoise(noise_pred_uncond, latents, time)
    x0_text = _to_denoise(noise_pred_text, latents, time)

    diff = x0_text - x0_uncond

    if APG_parameters["momentum_buffer"] is not None:
        APG_parameters["momentum_buffer"].update(diff)
        diff = APG_parameters["momentum_buffer"].running_avg

    if APG_parameters["norm_threshold"] > 0.0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim = [-1,-2,-3], keepdim=True)
        scale_factor = torch.minimum(ones, APG_parameters["norm_threshold"] / (diff_norm + 1e-8))
        diff = diff * scale_factor

    diff_parallel, diff_orthogonal = _project(diff, x0_uncond)

    normalized_update = diff_orthogonal+APG_parameters["eta"]*diff_parallel
    x0_guided = x0_text + (guidance_scale - 1) * normalized_update

    pred_guided = _to_noise(x0_guided, latents, time)

    return pred_guided

def rectified_pp_guidance(noise_pred_uncond, noise_pred_text, guidance_scale, time, latents, rectified_parameters):
    """
    Placeholder for the Rectified++ guidance method.
    """
    return NotImplementedError("Rectified++ guidance method is not implemented yet.")