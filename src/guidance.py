implemented_guidance = ["constant_guidance"]

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
    return noise_pred_uncond + (noise_pred_text - noise_pred_uncond) * guidance_scale