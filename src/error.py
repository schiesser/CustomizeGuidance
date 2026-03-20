from pathlib import Path

implemented_generative_model = ["SD3"]
implemented_performance_metrics = ["FID", "CLIP", "IS", "BLIP"]
implemented_guidance_methods = ["constant_guidance"]

def check_existing_generative_model(model_name: str):
    """Raises ValueError if the generative model is not implemented."""
    if model_name not in implemented_generative_model:
        raise ValueError(f"Model '{model_name}' not implemented. Available: {implemented_generative_model}.")

def check_existing_guidance_method(guidance_method_name: str):
    """Raises ValueError if the guidance method is not implemented."""
    if guidance_method_name not in implemented_guidance_methods:
        raise ValueError(f"Guidance method '{guidance_method_name}' not implemented. Available: {implemented_guidance_methods}.")

def check_existing_evaluation_metric(metric_name: str):
    """Raises ValueError if the evaluation metric is not implemented."""
    if metric_name not in implemented_performance_metrics:
        raise ValueError(f"Metric '{metric_name}' not implemented. Available: {implemented_performance_metrics}.")

def check_model_downloaded_path(model_path: str):
    """Raises FileNotFoundError if the model path does not exist."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"No model found at: '{model_path}'.")
    
def check_existing_data_path(data_folder_path: str):
    """Raises FileNotFoundError if the data folder path does not exist."""
    if not Path(data_folder_path).exists():
        raise FileNotFoundError(f"No data folder found at: '{data_folder_path}'.")