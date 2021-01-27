"""Create web app for uploading pictures."""
# https://medium.com/dev-genius/a-simple-way-to-build-flask-file-upload-1ccb9462bc2c

import logging
import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for
from rq import Queue
from werkzeug.utils import secure_filename

from src.core.data_utils import map_carbon_footprint
from src.main_script import execute_object_detection_script
from worker import conn

app = Flask(__name__)
q = Queue(connection=conn)

app.secret_key = "secret key"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

path = os.getcwd()
# file Upload
UPLOAD_FOLDER = os.path.join(path, "uploads")

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


ALLOWED_EXTENSIONS = set(["txt", "pdf", "png", "jpg", "jpeg", "gif"])

if __name__ != "__main__":
    gunicorn_logger = logging.getLogger("gunicorn.error")
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def upload_form():
    """Show upload form."""
    return render_template("upload.html")


@app.route("/", methods=["POST"])
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
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            flash("File successfully uploaded")
            file.seek(0)

            # Execute object detection and carbon mapping
            execute_object_detection_script(file)

            return redirect(url_for("get_user_input"))

        else:
            flash("Allowed file types are txt, pdf, png, jpg, jpeg, gif")
            return redirect(request.url)


@app.route("/input", methods=["GET", "POST"])
def get_user_input():
    if request.method == "POST":
        return redirect(url_for("script_output"))

    with open("src/temp_df.pickle", "rb") as file:
        output = pickle.load(file)

    product_list = output["detected_object"].unique()
    message = "Please enter amount"

    return render_template("input.html", message=message, contacts=product_list)


@app.route("/output", methods=["GET", "POST"])
def script_output():
    """Execute main script and return dataframe in html format."""

    req = request.form

    for key in req.keys():
        if key == "Amount":
            amount = pd.Series(req.getlist(key))
        if key == "Count":
            count = pd.Series(req.getlist(key))

    with open("src/temp_df.pickle", "rb") as file:
        output = pickle.load(file)

    output["amount"] = amount
    output["measurement"] = count

    # Create carbon df
    carbon_df = map_carbon_footprint(output)

    os.remove("src/temp_df.pickle")

    # Calculate overall carbon emission
    summed_carbon_emission = carbon_df[
        carbon_df[["total_carbon_emission"]]
        .applymap(lambda x: isinstance(x, (int, float)))
        .all(1)
    ].total_carbon_emission.sum()

    message = f"Your total carbon emission is: {summed_carbon_emission}g"

    carbon_df.fillna("No Value found", inplace=True)

    return render_template(
        "output.html", tables=[carbon_df.to_html(classes="footprint")], message=message
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
