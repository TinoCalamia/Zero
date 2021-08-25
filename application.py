"""Create web app for uploading pictures."""
# https://medium.com/dev-genius/a-simple-way-to-build-flask-file-upload-1ccb9462bc2c

import logging
import os
import pickle
import subprocess
import time

import numpy as np
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from src.core.data_utils import map_carbon_footprint
from src.main_script import execute_object_detection_script

application = Flask(__name__)

application.secret_key = "secret key"
application.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

path = os.getcwd()
# file Upload
UPLOAD_FOLDER = os.path.join(path, "uploads")

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

application.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


ALLOWED_EXTENSIONS = set(["txt", "pdf", "png", "jpg", "jpeg", "gif"])


if __name__ == "__main__":
    gunicorn_logger = logging.getLogger("gunicorn.error")
    application.logger.handlers = gunicorn_logger.handlers
    application.logger.setLevel(gunicorn_logger.level)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@application.route("/")
def upload_form():
    """Show upload form."""
    return render_template("upload.html")


@application.route("/", methods=["POST"])
def upload_file():
    """Upload file."""
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No file selected for uploading")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # Save file to folder
            filename = secure_filename(file.filename)
            file.save(os.path.join(application.config["UPLOAD_FOLDER"], filename))
            flash("File successfully uploaded")
            file.seek(0)

            # Execute object detection and carbon mapping
            # job = q.enqueue(execute_object_detection_script, file)
            execute_object_detection_script(file)

            return redirect(url_for("get_user_input"))

        else:
            flash("Allowed file types are txt, pdf, png, jpg, jpeg, gif")
            return redirect(request.url)


@application.route("/input", methods=["GET", "POST"])
def get_user_input():

    if request.method == "POST":
        return redirect(url_for("script_output"))

    with open("src/temp_df.pickle", "rb") as file:
        output = pickle.load(file)

    product_list = output["detected_object"].unique()
    message = "Please enter amount"

    return render_template("input.html", message=message, contacts=product_list)


@application.route("/output", methods=["GET", "POST"])
def script_output():
    """Execute main script and return dataframe in html format."""

    req = request.form

    for key in req.keys():
        if key == "Amount":
            amount = pd.Series(req.getlist(key))
        if key == "Count":
            count = pd.Series(req.getlist(key))
        if key == "Food":
            food = req.getlist(key)

    with open("src/temp_df.pickle", "rb") as file:
        temp_output = pickle.load(file)

    try:
        extended_object_list = temp_output["detected_object"].tolist() + food
    except Exception as e:
        print(f"Exception {e} was caused")
        extended_object_list = temp_output["detected_object"].tolist()

    output = pd.DataFrame()
    output["detected_object"] = pd.Series(extended_object_list)

    output["amount"] = amount
    output["measurement"] = count
    # Create carbon df
    carbon_df = map_carbon_footprint(output)

    os.remove("src/temp_df.pickle")

    # Calculate overall carbon emission
    summed_carbon_emission = round(
        carbon_df[
            carbon_df[["total_carbon_emission"]]
            .applymap(lambda x: isinstance(x, (int, float)))
            .all(1)
        ].total_carbon_emission.sum(),
        2,
    )

    message = f"Your total carbon emission is: {summed_carbon_emission}g"

    carbon_df.fillna("No Value found", inplace=True)

    return render_template(
        "output.html",
        tables=[
            carbon_df[
                [
                    "detected_object",
                    "amount",
                    "measurement",
                    "kg_carbon_per_unit",
                    "total_carbon_emission",
                ]
            ].to_html(classes="footprint")
        ],
        message=message,
    )

if __name__ == "__main__":
    #port = int(os.environ.get("PORT", 5000))
    application.run(host="0.0.0.0", debug=True)#, port=port
