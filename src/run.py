from .pipeline import StableDiffusion3PipelineCustomGuidance, Flux2KleinPipelineCustomGuidance
from .error import *
import torch
from .data_utils import extract_image_info
from .performance import compute_fid, compute_is, compute_clip_score, compute_blip_score
from tqdm import tqdm
import pickle
import shutil

def load_model(model: str, model_path: str, guidance_type: str, guidance_params: dict = None):
    """
    Load a generative model with a given guidance method.

    Args:
        model (str): Generative model to use. Available: ['SD3'].
        model_path (str): Path to the downloaded model.
        guidance_type (str): Guidance method to use.

    Returns:
        Loaded model pipeline.
    """
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    check_model_downloaded_path(model_path)
    check_existing_generative_model(model)
    check_existing_guidance_method(guidance_type)

    if model == "SD3":
        pipeline = StableDiffusion3PipelineCustomGuidance.from_pretrained(model_path, torch_dtype=torch.float32)
    
    if model == "FLUX2Klein":
        pipeline = Flux2KleinPipelineCustomGuidance.from_pretrained(model_path, torch_dtype=torch.float32)
    
    pipeline.configure_guidance(guidance_type=guidance_type, guidance_params=guidance_params)
    pipeline.to(torch_device)

    return pipeline

def generate_image(model_pipeline, prompt: str, height: int = 512, width: int = 512,
                   num_inference_steps: int = 28, guidance_scale: float = 7):
    """
    Generate an image using a loaded model pipeline.

    Args:
        model_pipeline: Loaded generative model pipeline.
        prompt (str): Text prompt for image generation.
        height (int): Height of the generated image in pixels.
        width (int): Width of the generated image in pixels.
        num_inference_steps (int): Number of denoising steps.
        guidance_scale (float): Adherence to the prompt vs. image quality.

    Returns:
        PIL.Image: Generated image.
    """
    result = model_pipeline(prompt=prompt, height=height, width=width, num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale, generator=torch.Generator().manual_seed(13))

    return result.images[0]

def run(model: str, guidance_type: str, model_path: str, prompt: str, height: int = 512, 
        width: int = 512, num_inference_steps: int = 28, guidance_scale: float = 7, 
        guidance_params: dict = None):
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
    model = load_model(model, model_path, guidance_type, guidance_params=guidance_params)

    generated_image = generate_image(model, prompt, height, width, num_inference_steps, guidance_scale)

    return generated_image

def benchmark(model: str, guidance_types: list[str], model_path: str, data_annotations_path: str, 
              data_images_path: str, num_inference_steps: int = 28, guidance_scale: float = 7, 
              score_list: list[str] = ["FID"], number_of_images: int = 5000, run_id: str = "test_run", 
              clip_model_path: str = None, blip_model_path: str = None, seed: int = 13, height:int=512,
              width:int=512, guidance_parameters: list[dict] = None, keep_images: bool = False):
    """
    Run a benchmark:
    retrieve scores for guidances_types for a given generative model and a given dataset.
    The way the prompt are retrieved work for MS-COCO dataset.
    For other dataset the function "extract_image_info" need to be changed.

    Args:
        model (str): Generative model to use. Available: ['SD3'].
        guidance_types (list[str]): List of guidance methods to use.
        model_path (str): Path to the downloaded model.
        data_annotations_path (str): Path to the dataset annotations (csv file).
        data_images_path (str): Path to the dataset images (folder).
        height (int): Height of the generated image in pixels.
        width (int): Width of the generated image in pixels.
        num_inference_steps (int): Number of denoising steps.
        guidance_scale (float): Adherence to the prompt vs. image quality.
        score_list (list[str]): List of evaluation metrics to compute. Available: ['FID', 'IS', 'CLIP', 'BLIP'].
        number_of_images (int): Number of images to generate and evaluate. (from COCO caption, max 5000)
        run_id (str): Id of the run, used to save generated images in a specific folder.
        clip_model_path (str): Path to the CLIP model to be used for scoring. Required if "CLIP" in score_list.
        blip_model_path (str): Path to the BLIP model to be used for scoring. Required if "BLIP" in score_list.
    
    Returns:
        dict: Dictionary containing the scores for each guidance method and each evaluation metric.
    """
    # Validate inputs
    check_model_downloaded_path(model_path)
    check_existing_generative_model(model)
    for guidance_method in guidance_types: check_existing_guidance_method(guidance_method)
    check_existing_data_path(data_annotations_path)
    check_existing_data_path(data_images_path)
    for score_name in score_list: check_existing_evaluation_metric(score_name)

    # get captions, dimensions and jpeg name of the original images
    images_info = extract_image_info(data_annotations_path, seed=seed, keep_divisible_16 = True)
    
    # generate images with the given model/guidance method for every prompt and save them in a target folder 
    full_score = {}
    for i, guidance_method in tqdm(enumerate(guidance_types), total=len(guidance_types), desc="Guidance methods"):

        if any(s in score_list for s in ["IS", "CLIP", "BLIP"]):
            path_generated_images = f"outputs/{run_id}/generated_images/{guidance_method}"
            Path(path_generated_images).mkdir(parents=True, exist_ok=True)

        if "FID" in score_list:
            path_original_fid = f"outputs/{run_id}/FID/original_images/{guidance_method}"
            Path(path_original_fid).mkdir(parents=True, exist_ok=True)
            path_generated_fid = f"outputs/{run_id}/FID/generated_images/{guidance_method}"
            Path(path_generated_fid).mkdir(parents=True, exist_ok=True)

        pipeline_model = load_model(model, model_path, guidance_method, guidance_parameters[i] if guidance_parameters is not None else None)

        for _, row in tqdm(images_info.iloc[:number_of_images].iterrows(), total=number_of_images, desc=f"    [{guidance_method}] Generating images"):

            # for FID (keep same dimension between original and generated images)
            if "FID" in score_list:
                pipeline_model.guidance_method.reset()
                generated_image_fid = generate_image(pipeline_model, row['caption'], row['height'], row['width'], num_inference_steps, guidance_scale)
                generated_image_fid.save(f"{path_generated_fid}/{row['file_name']}")
                shutil.copy(f"{data_images_path}/{row['file_name']}", f"{path_original_fid}/{row['file_name']}")

            if any(s in score_list for s in ["IS", "CLIP", "BLIP"]):
                pipeline_model.guidance_method.reset()
                generated_image = generate_image(pipeline_model, row['caption'], height, width, num_inference_steps, guidance_scale)
                generated_image.save(f"{path_generated_images}/{row['file_name']}")
        
        # evaluate the score of the generated images against the original ones with the given evaluation metric(s)
        dict_score = {}
        fid_score, is_mean, is_std, clip_score, blip_score = None, None, None, None, None

        # get prompts and order the prompts wrt the "file_name" features
        prompts = images_info['caption'].iloc[:number_of_images].tolist()
        sorted_key = images_info['file_name'].iloc[:number_of_images].tolist()
        prompts = [prompt for _, prompt in sorted(zip(sorted_key, prompts))]

        # compute score
        if "FID" in score_list:
            fid_score = compute_fid(path_generated_fid, path_original_fid)
        if "IS" in score_list :
            is_mean, is_std = compute_is(path_generated_images, seed=seed)   
        if "CLIP" in score_list :
            clip_score = compute_clip_score(path_generated_images, prompts, clip_model_path)
        if "BLIP" in score_list:
            blip_score = compute_blip_score(path_generated_images, prompts, blip_model_path)

        dict_score["fid"] = fid_score
        dict_score["is_mean"] = is_mean
        dict_score["is_std"] = is_std
        dict_score["clip"] = clip_score
        dict_score["blip"] = blip_score

        full_score[guidance_method] = dict_score

    if not keep_images:
        shutil.rmtree(f"outputs/{run_id}")

    save_file = open(run_id, 'wb')
    pickle.dump(full_score, save_file)
    save_file.close()

    return full_score
