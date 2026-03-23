from pathlib import Path

implemented_generative_model = ["SD3"]
implemented_performance_metrics = ["FID", "CLIP", "IS", "BLIP"]
implemented_guidance_methods = ["constant", "linear", "exponential", "APG"]
REQUIRED_APG_PARAMETERS = {"momentum_value": (int, float),"norm_threshold": (int, float),
                           "eta": (int, float),"momentum_buffer": (type(None), object)}

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
    
def check_guidance_parameters(guidance_type: str, guidance_params: dict | None):
    """
    Validate guidance parameters depending on the selected guidance type.

    For APG and rectified_pp, ensures:
        - guidance_params is provided
        - required keys are present
        - values have correct types

    Raises:
        ValueError: if parameters are missing or invalid
    """

    if guidance_type in ["constant", "linear", "exponential"]:
        return

    # Ensure dict is provided
    if guidance_params is None:
        raise ValueError(f"`guidance_params` must be provided for guidance_type='{guidance_type}'")

    if not isinstance(guidance_params, dict):
        raise TypeError(f"`guidance_params` must be a dict, got {type(guidance_params)}")

    # APG
    if guidance_type == "APG":
        required_keys = {"momentum_value": float, "eta": float, "norm_threshold": float}

        for key, expected_type in required_keys.items():
            if key not in guidance_params:
                raise ValueError(f"Missing key '{key}' in APG parameters")

            if not isinstance(guidance_params[key], expected_type):
                raise TypeError(f"APG parameter '{key}' must be of type {expected_type}, got {type(guidance_params[key])}")

    # Rectified++
    elif guidance_type == "rectified_pp":
        required_keys = {"alpha_mode": str}

        optional_keys = {"eta": float,"dt_scale": float}

        for key, expected_type in required_keys.items():
            if key not in guidance_params:
                raise ValueError(f"Missing key '{key}' in rectified_pp parameters")

            if not isinstance(guidance_params[key], expected_type):
                raise TypeError(f"rectified_pp parameter '{key}' must be of type {expected_type}, got {type(guidance_params[key])}")
            
        for key, expected_type in optional_keys.items():
            if key in guidance_params and not isinstance(guidance_params[key], expected_type):
                raise TypeError(f"rectified_pp parameter '{key}' must be of type {expected_type}, got {type(guidance_params[key])}")


    
