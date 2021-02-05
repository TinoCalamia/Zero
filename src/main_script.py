"""Run prediction for image."""
import os
import pickle

import tensorflow as tf
import tensorflow_hub as hub

from src.core.data_utils import (
    get_results_with_score,
    get_unique_objects,
    map_carbon_footprint,
    run_detector,
    time_it,
)
from src.core.image_utils import resize_image
from src.utils.setup import object_detection_setup_config as setup


@time_it
def load_model():
    # Load model
    if os.path.exists(
        os.path.join("src", setup.model_dir, setup.model_name, "saved_model.pb")
    ):
        print("Directory is not empty. Use local model.")
        detector = tf.saved_model.load(
            os.path.join("src", setup.model_dir, setup.model_name)
        ).signatures["default"]

    else:
        print("Directory is empty. Load model from Tensorflow Hub")
        detector = hub.load(setup.module_handle).signatures["default"]

    return detector


detector = load_model()


def execute_object_detection_script(file, detector=detector):
    """Execute all steps."""

    # image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Fruits_veggies.png/1200px-Fruits_veggies.png"
    image_path = resize_image(file, 1280, 856, False)

    print("Start execution.")
    # Make detection
    detected_objects_dict = run_detector(
        detector=detector, image_path=image_path, show_image=False
    )
    # Make df with objects and scores
    detection_df = get_results_with_score(detected_objects_dict)
    # Create df with unique items
    unique_detected_objects = get_unique_objects(detection_df)

    with open("src/temp_df.pickle", "wb") as file:
        pickle.dump(unique_detected_objects, file)

    print("Finished execution.")
