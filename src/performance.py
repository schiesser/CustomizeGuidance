from cleanfid import fid
import torch_fidelity
import torch
from torchmetrics.multimodal import CLIPScore
from PIL import Image
from pathlib import Path
import numpy as np
from transformers import BlipProcessor, BlipForImageTextRetrieval

def compute_fid(generated_image_path: str,
                real_image_path: str):
    """
    Computes the FID score between generated and real images.

    Args:
        generated_image_path (str): Path to the directory containing generated images.
        real_image_path (str): Path to the directory containing real images.
    """
    score = fid.compute_fid(generated_image_path, real_image_path)
    return score

def compute_is(generated_image_path: str, 
               seed:int = 13):
    """
    Computes the Inception Score (IS) for generated images.

    Args:
        generated_image_path (str): Path to the directory containing generated images.
    """
    metrics = torch_fidelity.calculate_metrics(input1=generated_image_path, isc=True, isc_splits=10, rng_seed=seed)
    mean = metrics["inception_score_mean"]
    std = metrics["inception_score_std"]
    return mean, std

def compute_clip_score(generated_image_path: str, 
                       prompts: list[str], 
                       clip_model_path: str):
    """
    Computes the CLIP score between generated images and their corresponding prompts.
    This function used images that have been generated and saved in a folder.
    For better performance the score can be computed on images one by one after generation, 
    a step of open can be avoided. But here we favorize the "workflow".

    Args:
        generated_image_path (str): Path to the directory containing generated images.
        prompts (list[str]): List of prompts corresponding to the generated images.
        clip_model_path (str): Path to the CLIP model to be used for scoring.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip = CLIPScore(model_name_or_path=clip_model_path)
    clip.to(device)
    
    images = [torch.tensor(np.array(Image.open(p).convert("RGB"))).permute(2, 0, 1) for p in sorted(Path(generated_image_path).glob("*.png"))]
    score = clip(images, prompts)
    return score.item()

def compute_blip_score(generated_image_path: str, 
                       prompts: list[str]):
    """
    Computes the BLIP score between generated images and their corresponding prompts.
    This function used images that have been generated and saved in a folder.
    (comments about performance are as for CLIP score)

    Args:
        generated_image_path (str): Path to the directory containing generated images.
        prompts (list[str]): List of prompts corresponding to the generated images. 

    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
    model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
    model.to(device)
    model.eval()

    scores = []
    image_paths = sorted(Path(generated_image_path).glob("*.png"))

    for path, prompt in zip(image_paths, prompts):
        image = Image.open(path).convert("RGB")
        inputs = processor(image, prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            score = model(**inputs, use_itm_head=True).itm_score
            score = torch.nn.functional.softmax(score, dim=1)[:, 1].item()

        scores.append(score)

    mean_score = sum(scores) / len(scores)
    return mean_score