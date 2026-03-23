import numpy as np

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