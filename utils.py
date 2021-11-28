import numpy as np
import os
import cv2
import logging
logger = logging.getLogger("main")



COLORS = np.random.uniform(0, 255, size=(80, 3))
classes = None


def populate_class_labels():
    class_file_name = "models/yolov3_classes.txt"
    f = open(class_file_name, "r")
    classes = [line.strip() for line in f.readlines()]
    return classes


# function to calculate dynamic font and thickness
def dynamic_font(frame):
    """
    :objective: take the input frame and calculate suitable font
         size and thickness

    :param frame: currnt input frame

    returns:
    :param fontscale , thickness

    """

    (h, w) = frame.shape[:2]

    # above 120 lacs
    if h * w >= 12000011:
        fontscale = 3.5
        thickness = 13

    # between 50 to 120lacs
    elif 5000000 <= h * w <= 12000000:
        fontscale = 2.5
        thickness = 10

    # between 20 to 50lacs
    elif 2000001 <= h * w <= 4900000:
        fontscale = 2
        thickness = 8

    # between 12 to 20lacs
    elif 1200000 <= h * w <= 2000000:
        fontscale = 1.5
        thickness = 7

    # between 5 to 12lacs
    elif 500000 <= h * w <= 1200001:
        fontscale = 1
        thickness = 4

    # between 3 to 5lacs
    elif 300001 <= h * w <= 499999:
        fontscale = 0.80
        thickness = 4

    # between 1 to 3lacs
    elif 100001 <= h * w <= 300000:
        fontscale = 0.50
        thickness = 2

    # less than 1lacs
    elif h * w <= 100000:
        fontscale = 0.25
        thickness = 1

    # defualt
    else:
        fontscale = 0.45
        thickness = 2

    return fontscale, thickness


def annotate(img, bbox, labels, confidence, colors=None, write_conf=False):
    global COLORS
    global classes

    if classes is None:
        classes = populate_class_labels()

    # calculate dynamic font
    fontscale, thickness = dynamic_font(img)

    for i, label in enumerate(labels):

        if colors is None:
            color = COLORS[classes.index(label)]
        else:
            color = colors[classes.index(label)]

        if write_conf:
            label += " " + str(format(confidence[i] * 100, ".2f")) + "%"

        cv2.rectangle(img, (bbox[i][0], bbox[i][1]), (bbox[i][2], bbox[i][3]), color, thickness)
        cv2.putText(img, label, (bbox[i][0], bbox[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, fontscale, color, thickness)

    return img
