"""Run prediction for image."""
import tensorflow_hub as hub

from core.data_utils import (
    get_results_with_score,
    get_unique_objects,
    map_carbon_footprint,
    run_detector,
)
from core.image_utils import download_and_resize_image
from utils.setup import object_detection_setup_config as setup

image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Fruits_veggies.png/1200px-Fruits_veggies.png"
downloaded_image_path = download_and_resize_image(image_url, 1280, 856, False)

# Load model
detector = hub.load(setup.module_handle).signatures["default"]

if __name__ == "__main__":
    # Make detection
    detected_objects_dict = run_detector(detector, downloaded_image_path, False)
    # Make df with objects and scores
    detection_df = get_results_with_score(detected_objects_dict)
    # Create list with unique items
    unique_detected_objects = get_unique_objects(detection_df)
    # Create carbon df
    carbon_df = map_carbon_footprint(unique_detected_objects)
    print(carbon_df)
    # Calculate ghg based on piece or weight
    ##
