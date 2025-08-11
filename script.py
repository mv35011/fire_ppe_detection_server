import json
import concurrent.futures
from typing import Dict, Any, List

# Import model modules
from models import fire_detection
from models import ppe_detection
from models import number_plate_detection
from models import fall_detection
from utils.utils import decode_request_image

# Model mapping
MODEL_PROCESSORS = {
    'fire': fire_detection,
    'ppe': ppe_detection,
    'number_plate': number_plate_detection,
    'fall': fall_detection
}

def init() -> None:
    """Initialize all models."""
    fall_detection.initialize()
    fire_detection.initialize()
    ppe_detection.initialize()
    number_plate_detection.initialize()

def process_with_model(image_bytes: bytes, model_type: str) -> Dict[str, Any]:
    """
    Process an image with the specified model type.
    
    Args:
        image_bytes: Raw image bytes
        model_type: The model to use (fire, ppe, or number_plate)
        
    Returns:
        Dictionary with detection results
    """
    if model_type in MODEL_PROCESSORS:
        return MODEL_PROCESSORS[model_type].process(image_bytes)
    else:
        return {"status": "error", "message": f"Invalid model type: {model_type}", "model": model_type}

def run(raw_data: str) -> str:
    """
    Process incoming request data with multiple models in parallel.
    
    Args:
        raw_data: Raw JSON request data
        
    Returns:
        JSON string with combined results
    """
    try:
        image_bytes, model_types = decode_request_image(raw_data)
        
        # Filter to only use supported models
        valid_model_types = [m for m in model_types if m in MODEL_PROCESSORS]
        if not valid_model_types:
            valid_model_types = ['fire']  # Default to fire if no valid models
        
        combined_results = []
        
        # Process all requested models in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_with_model, image_bytes, model_type) 
                for model_type in valid_model_types
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                combined_results.append(result)

        return json.dumps({"results": combined_results})

    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


if __name__ == "__main__":
    # Initialize all models on startup
    init()
    
    # Example: You can add any server code here that calls the run function
    print("Models initialized and ready for inference")