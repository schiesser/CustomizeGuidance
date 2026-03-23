import numpy as np
import torch

def constant_guidance(noise_pred_uncond, noise_pred_text, guidance_scale):
    """
    Applies constant guidance to the noise prediction.
    
    Args:
        noise_pred_uncond: The noise prediction for the unconditional input.
        noise_pred_text: The noise prediction for the text input.
        guidance_scale: The scale of the guidance to apply.
    
    Returns:
        The guided noise prediction.
    """
    omega = guidance_scale
    return noise_pred_uncond + (noise_pred_text - noise_pred_uncond) * omega

def linear_guidance(noise_pred_uncond, noise_pred_text, guidance_scale, current_step, total_steps):
    """ 
    Applies increasing linear guidance to the noise prediction.
    It satisfies: 
        integral from 0 to T of omega(t) = guidance_scale * T.
    
    Args:
        noise_pred_uncond: The noise prediction for the unconditional input.
        noise_pred_text: The noise prediction for the text input.
        guidance_scale: The maximum scale of the guidance to apply.
        current_step: The current inference step.
        total_steps: The total number of inference steps.
    
    Returns:
        The guided noise prediction.
    """
    # Calculate the linear scaling factor based on the current step
    omega = 2 * (1 - current_step / total_steps) * guidance_scale
    return noise_pred_uncond + (noise_pred_text - noise_pred_uncond) * omega

def exponential_guidance(noise_pred_uncond, noise_pred_text, guidance_scale, time, t_max):
    """ 
    Applies increasing exponential guidance to the noise prediction.
    It satisfies: 
        integral from 0 to T of omega(t) = guidance_scale * T.
    
    Args:
        noise_pred_uncond: The noise prediction for the unconditional input.
        noise_pred_text: The noise prediction for the text input.
        guidance_scale: The maximum scale of the guidance to apply.
        time: The current time step.
        t_max: The maximum time step. 
            time < t_max !
    
    Returns:
        The guided noise prediction.
    """
    # Calculate the exponential scaling factor based on the current step
    alpha = (guidance_scale*t_max + 1 - np.exp(t_max)) / (- t_max)
    omega = alpha * (np.exp(time))
    return noise_pred_uncond + (noise_pred_text - noise_pred_uncond) * omega


def adaptative_projected_guidance(noise_pred_uncond, noise_pred_text, guidance_scale, time, APG_parameters, latents):
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

    diff_parallel, diff_orthogonal = _project(diff, noise_pred_uncond)

    normalized_update = diff_orthogonal+APG_parameters["eta"]*diff_parallel
    pred_guided = noise_pred_text + (guidance_scale -1) * normalized_update

    _to_noise(pred_guided, latents, time)

    return pred_guided

def _project(v0, v1):
    """
    Projects v0 onto v1 and computes the orthogonal and parallel components.

    Args:
        v0: The vector to be projected, shape (B, C, H, W)
        v1: The vector to project onto, shape (B, C, H, W)

    Returns:
        v0_parallel: The component of v0 parallel to v1, shape (B, C, H, W)
        v0_orthogonal: The component of v0 orthogonal to v1, shape (B, C, H, W)
    """
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim = [-1,-2,-3])
    v0_parallel = (v0 * v1).sum(dim = [-1,-2,-3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)

def _to_denoise(v_t, x_t, t):
    """
    Estimate denoised image from current noisy latents and velocity prediction.
    Flow matching : x0 = x_t - t * v_t
    
    Args:
        v_t : velocity prediction (noise_pred), shape (B, C, H, W)
        x_t : current noisy latents, shape (B, C, H, W)
        t   : current timestep scalar ∈ [0, 1]
    
    Returns:
        x0 : denoised estimate, shape (B, C, H, W)
    """
    return x_t - t * v_t


def _to_noise(x0, x_t, t):
    """
    Estimate noise from current noisy latents and denoised estimate.
    Flow matching : v_t = (x_t - x0) / t
    
    Args:
        x0  : denoised estimate, shape (B, C, H, W)
        x_t : current noisy latents, shape (B, C, H, W)
        t   : current timestep scalar ∈ [0, 1]

    Returns:
        v_t : velocity prediction (noise_pred), shape (B, C, H, W)
    """
    if t < 1e-6:
        return x0
    return (x_t - x0) / t