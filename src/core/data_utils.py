"""Utility functions for data processing."""
import cv2
import json
import os
import pickle
import subprocess
import time
import uuid

import gspread
import numpy as np
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials

from src.core.image_utils import display_image, draw_boxes
from src.core.nlp_utils import singularize_words
from src.utils.setup import object_detection_setup_config as setup


def time_it(f):
    """Time function."""

    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print(f"func {f.__name__} took: {te - ts: 2.4f} seconds")
        return result

    return timed


def write_image_file(name, path):
    f = open(name, "w+")
    f.write(path)
    f.close()


@time_it
def run_detector(image_path, show_image=False, thresh=0.0001):
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
    # define network
    net = cv2.dnn.readNetFromDarknet(
        "yolov4/yolov4-tiny-custom.cfg",
        "yolov4/backup/Fruits_Best.weights",
    )

    image = cv2.imread(image_path)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)
    start_time = time.time()
    layerOutputs = net.forward(ln)
    end_time = time.time()

    output_result = pd.DataFrame()
    detected_classes = []
    confidences = []

    # Code example
    # https://cloudxlab.com/blog/object-detection-yolo-and-python-pydarknet/
    LABELS = open("yolov4/obj.names").read().strip().split("\n")

    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            confidence = scores[np.argmax(scores)]
            label = LABELS[np.argmax(scores)]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > 0.1:
                # update our list confidences,
                # and class IDs
                confidences.extend([float(confidence)])
                detected_classes.extend([label])

    output_result["detected_objects"] = pd.Series(detected_classes)
    output_result["scores"] = pd.Series(confidences)

    print("Inference time: ", end_time - start_time)
    return output_result


@time_it
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
    entities_decoded = np.array([x for x in result["detected_objects"]])
    rounded_scores = np.array([round(x, 3) for x in result["scores"]])
    result_df[object_column] = entities_decoded
    result_df[target_column] = rounded_scores

    print(result_df.sort_values(by=target_column, ascending=False).head(10))

    return result_df.sort_values(by=target_column, ascending=False)


@time_it
def get_unique_objects(
    dataframe, object_column="object", target_column="score", threshold=0.2
):
    """Get list with unique objects above threshold."""
    return (
        dataframe[dataframe[target_column] > threshold][[object_column]]
        .drop_duplicates()
        .rename(columns={object_column: "detected_object"})
    )


@time_it
def map_carbon_footprint(output):
    """Map carbon data to detected objects."""
    # Pre-process ghg data
    ghg_data = get_df_from_spreadsheet("ghg_data")
    # convert columns to float
    for column in [
        "Land use change",
        "Animal Feed",
        "Farm",
        "Processing",
        "Transport",
        "Packging",
        "Retail",
    ]:
        ghg_data[column] = ghg_data[column].astype(float)

    ghg_data["total_emission"] = ghg_data.iloc[:, 1:].sum(axis=1)
    # Convert Words to singular
    ghg_data["Food product"] = np.array(singularize_words(ghg_data["Food product"]))

    # Create new dataframe
    foodprint_df = output.copy(deep=True)
    # Merge carbon foodprint
    foodprint_df = foodprint_df.merge(
        ghg_data[["Food product", "total_emission"]],
        left_on="detected_object",
        right_on="Food product",
        how="left",
    ).drop("Food product", axis=1)
    foodprint_df.rename(columns={"total_emission": "kg_carbon_per_unit"}, inplace=True)

    # Add average weight per product to calculate emission per "Piece"
    weight_df = get_df_from_spreadsheet("food_weight_data")
    weight_df.avg_weight_per_piece = weight_df.avg_weight_per_piece.astype(float)
    foodprint_df = foodprint_df.merge(
        weight_df,
        how="left",
        left_on="detected_object",
        right_on="product",
    ).drop("product", axis=1)

    # Calculate total carbon emission
    foodprint_df["total_carbon_emission"] = np.where(
        foodprint_df.measurement == "Kg",
        foodprint_df.amount.astype(float)
        * foodprint_df.kg_carbon_per_unit.astype(float),
        np.where(
            foodprint_df.measurement == "Piece",
            (foodprint_df.avg_weight_per_piece / 1000)
            * foodprint_df.kg_carbon_per_unit.astype(float).astype(float),
            np.nan,
        ),
    )

    return foodprint_df


def get_df_from_spreadsheet(spreadsheet_name):
    """Generate data from spreadsheet."""
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]

    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        "src/zero-carbon-emission-ab5ea26380ab.json", scope
    )  # Your json file here

    gc = gspread.authorize(credentials)
    wks = gc.open(spreadsheet_name).sheet1

    data = wks.get_all_values()
    headers = data.pop(0)

    df = pd.DataFrame(data, columns=headers)

    return df


def get_class_string_from_index(index):
    with open("src/data/target_classes.pickle", "rb") as file:
        target_classes = pickle.load(file)

    for class_string, class_index in target_classes:
        if class_index == index:
            return class_string
