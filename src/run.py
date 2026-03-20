from pipeline import StableDiffusion3PipelineCustomGuidance
from guidance import implemented_guidance
import torch 
from pathlib import Path

def run(model: str,
        guidance_type: str,
        model_path: str,
        prompt:str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 28,
        guidance_scale: float = 7
        ):
    
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not Path(model_path).exists(): raise FileNotFoundError(f'No model at the location: {model_path}.')
    if guidance_type not in implemented_guidance: raise ValueError(f'{guidance_type} not implemented.')

    if model == "SD3":
        sd3 = StableDiffusion3PipelineCustomGuidance.from_pretrained(model_path, guidance_type)
        sd3.to(torch_device)

        result = sd3(prompt = prompt,
            height = height,
            width = width,
            num_inference_steps = num_inference_steps,
            guidance_scale=guidance_scale,
            generator = torch.Generator().manual_seed(13)
            )
    else:
        raise ValueError(f'Model {model} not implemented. Please choose from: SD3')
        
    return result.images[0]
