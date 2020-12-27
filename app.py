"""Create web app for uploading pictures."""
# https://medium.com/dev-genius/a-simple-way-to-build-flask-file-upload-1ccb9462bc2c

import logging
import os
import pickle
from rq import Queue
from worker import conn

from flask import Flask, flash, redirect, render_template, request
from werkzeug.utils import secure_filename
from src.main_script import execute_object_detection_script

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


@app.route("/uploaded")
def script_output():
    """Execute main script and return dataframe in html format."""

    with open("src/temp_df.pickle", "rb") as file:
        output = pickle.load(file)

    os.remove("src/temp_df.pickle")

    return render_template("output.html", tables=[output.to_html(classes="footprint")])


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

            # Execute object detection and carbon mapping
            execute_object_detection_script()

            #######
            # ASK USER FOR INPUT HERE
            #######

            # Return reults to user
            return script_output()

        else:
            flash("Allowed file types are txt, pdf, png, jpg, jpeg, gif")
            return redirect(request.url)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
