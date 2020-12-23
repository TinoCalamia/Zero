"""Utility functions for data processing."""
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from core.image_utils import display_image, draw_boxes, load_img
from core.nlp_utils import singularize_words
from utils.setup import object_detection_setup_config as setup


def run_detector(detector, path, show_image=False):
    """
    Run detector and return detection summary.

    Parameters;
    ----------
    path (str): String which contains the URL for downloading the image
    show_image (bool): If True image with detected objects will be displayed


    Returns;
    -------
    result (dict): Dictionary including detection results

    """
    img = load_img(path)
    converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    start_time = time.time()
    result = detector(converted_img)
    end_time = time.time()

    result = {key: value.numpy() for key, value in result.items()}

    print("Found %d objects." % len(result["detection_scores"]))
    print("Inference time: ", end_time - start_time)

    if show_image:
        image_with_boxes = draw_boxes(
            img.numpy(),
            result["detection_boxes"],
            result["detection_class_entities"],
            result["detection_scores"],
        )

        display_image(image_with_boxes)

    return result


def get_results_with_score(result, object_column="object", target_column="score"):
    """
    Convert detection dictionary to dataframe.

    Parameters;
    ----------
    result (dict): Dictionary containing Tensorflow detection summary
    object_column (str): Name of the column which contains the detected objects
    target_column (str): Name of the column which contains the scores for the detected objects


    Returns;
    -------
    result_df (dataframe): Dataframe containing objects and score of detection dictionary

    """

    result_df = pd.DataFrame()
    entities_decoded = np.array(
        [x.decode("utf-8") for x in result["detection_class_entities"]]
    )
    rounded_scores = np.array([round(x, 3) for x in result["detection_scores"]])
    result_df[object_column] = entities_decoded
    result_df[target_column] = rounded_scores

    return result_df.sort_values(by=target_column, ascending=False)


def get_unique_objects(
    dataframe, object_column="object", target_column="score", threshold=0.5
):
    """Get list with unique objects above threshold."""
    return dataframe[dataframe[target_column] > threshold][object_column].unique()


def map_carbon_footprint(unique_detected_objects):
    """Map carbon data to detected objects."""
    # Pre-process ghg data
    ghg_data = pd.read_csv(
        os.path.join("src", setup.data_dir, setup.ghg_data_file_name)
    ).drop("Unnamed: 8", axis=1)
    ghg_data.columns = setup.column_names
    ghg_data["total"] = ghg_data.iloc[:, 1:].sum(axis=1)
    # Convert Words to singular
    ghg_data["product"] = np.array(singularize_words(ghg_data["product"]))

    # Create new dataframe
    foodprint_df = pd.DataFrame()
    foodprint_df["detected_object"] = np.array(unique_detected_objects)
    foodprint_df["measurement"] = None
    foodprint_df["amount"] = None
    # Merge carbon foodprint
    foodprint_df = foodprint_df.merge(
        ghg_data[["product", "total"]],
        left_on="detected_object",
        right_on="product",
        how="left",
    )
    foodprint_df.rename(columns={"total": "kg_carbon_per_kg"}, inplace=True)

    return foodprint_df
