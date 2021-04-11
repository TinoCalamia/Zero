"""Utility functions for displaying images."""

import ssl
import tempfile

import certifi

# For downloading the image.
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps
from six import BytesIO
from six.moves.urllib.request import urlopen

from src.utils.setup import object_detection_setup_config as setup


def display_image(image):
    """Display image."""
    plt.figure(figsize=(20, 15))
    plt.grid(False)
    plt.imshow(image)


def resize_image(image_file, display=False):
    """
    Download image from web and resize to chosen specs.

    Parameters;
    ----------
    url (str): String which contains the URL for downloading the image
    new_width (int): Displayed width of the picture
    new_height (int): Displayed width of the picture
    display (bool): If True image will be displayed

    Returns;
    -------
    Resized picture.

    """
    image_data = image_file.stream.read()
    image_data = BytesIO(image_data)
    pil_image = Image.open(image_data)
    pil_image = ImageOps.fit(
        pil_image, (setup.image_width, setup.image_height), Image.ANTIALIAS
    )
    pil_image_rgb = pil_image.convert("RGB")
    if image_file.filename[-3:] == "png":
        format_option = "PNG"
    else:
        format_option = "JPEG"
    pil_image_rgb.save(image_file.filename, format=format_option, quality=90)
    print("Using image %s." % image_file.filename)

    if display:
        display_image(pil_image)

    return image_file.filename


def draw_bounding_box_on_image(
    image, ymin, xmin, ymax, xmax, color, font, thickness=4, display_str_list=()
):
    """Add a bounding box to an image."""
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (
        xmin * im_width,
        xmax * im_width,
        ymin * im_height,
        ymax * im_height,
    )
    draw.line(
        [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
        width=thickness,
        fill=color,
    )

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [
                (left, text_bottom - text_height - 2 * margin),
                (left + text_width, text_bottom),
            ],
            fill=color,
        )
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill="black",
            font=font,
        )
        text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    colors = list(ImageColor.colormap.values())

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf", 25
        )
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(
                class_names[i].decode("ascii"), int(100 * scores[i])
            )
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(
                image_pil,
                ymin,
                xmin,
                ymax,
                xmax,
                color,
                font,
                display_str_list=[display_str],
            )
            np.copyto(image, np.array(image_pil))
    return image
