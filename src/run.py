from pipeline import StableDiffusion3PipelineCustomGuidance
from error import check_existing_generative_model, check_existing_guidance_method, check_model_downloaded_path
import torch

def run(model: str,
        guidance_type: str,
        model_path: str,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 28,
        guidance_scale: float = 7
        ):
    """
    Run inference on a generative model with a given guidance method.

    Args:
        model (str): Generative model to use. Available: ['SD3'].
        guidance_type (str): Guidance method to use.
        model_path (str): Path to the downloaded model.
        prompt (str): Text prompt for image generation.
        height (int): Height of the generated image in pixels.
        width (int): Width of the generated image in pixels.
        num_inference_steps (int): Number of denoising steps.
        guidance_scale (float): Adherence to the prompt vs. image quality.

    Returns:
        PIL.Image: Generated image.
    """
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    check_model_downloaded_path(model_path)
    check_existing_generative_model(model)
    check_existing_guidance_method(guidance_type)

    if model == "SD3":
        sd3 = StableDiffusion3PipelineCustomGuidance.from_pretrained(model_path, guidance_type=guidance_type)
        sd3.to(torch_device)

        result = sd3(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(13)
        )

    return result.images[0]