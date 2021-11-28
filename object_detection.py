import cvlib as cv
import cv2
from cvlib.object_detection import draw_bbox
from config import *
from utils import *
import logging
logger = logging.getLogger("main")

if METHOD == "SSD":
    logger.debug("[INFO] SSD model used for object detection...")
elif METHOD == "YOLOV3":
    logger.debug("[INFO] Yolov3 model used for object detection...")



def detect_object_yolo(img):
    """
    :obj detect 80 objects from the input image and drow the bbox and write label with conf
    :param img: input image for processing
    :return: output image with bbox
    """
    bbox, label, conf = cv.detect_common_objects(img, confidence=CONF, model=MODEL, enable_gpu=ENABLE_GPU)
    output_image = annotate(img, bbox, label, conf, write_conf=W_CONF)
    logger.debug("[INFO] detected object using Yolov3: {}".format(label))
    return output_image


# load the caffe model
net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)
logger.debug("[INFO] MobileNet SSD model loaded successfully...")


def detect_object_ssd(frame):
    """
    :obj detect the 20 objects from the given image and draw the box with conf
    :param frame:
    :return: output image with bounding box
    """
    frame_resized = cv2.resize(frame, (300, 300))  # resize the frame for prediction
    # MobileNet requires fixed dimensions for input imsge(S)
    # so we have to ensure that it is resized to 300 x 300 pixels
    # sete a scale factor to image because network the objects has differts size
    # we perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
    # after executing this command our "blob" now has the shape
    # (1,3,300,300)

    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)

    # set to network the input blob
    net.setInput(blob)

    # prediction of network
    detections = net.forward()

    # size of frame resize (300 x 300)
    cols = frame_resized.shape[1]
    rows = frame_resized.shape[0]

    # for get the class and location of object detected,
    # there is fix index for class, location and confidence
    # value in @detections array
    objects = set()

    per_frame_object = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # confidence of prediction
        if confidence > CONF:  # filter prediciton
            class_id = int(detections[0, 0, i, 1])

            # object location
            xLeftBottom = int(detections[0, 0, i, 3] * cols)
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop = int(detections[0, 0, i, 5] * cols)
            yRightTop = int(detections[0, 0, i, 6] * rows)

            # Factor for scale to orignal size of frame
            heightFactor = frame.shape[0] / 300.0
            widhtFactor = frame.shape[1] / 300.0

            # sclae object detection to frame
            xLeftBottom = int(widhtFactor * xLeftBottom)
            yLeftBottom = int(heightFactor * yLeftBottom)
            xRightTop = int(widhtFactor * xRightTop)
            yRightTop = int(heightFactor * yRightTop)

            # calculate dynamic font
            fontscale, thickness = dynamic_font(frame)

            # draw location of object
            cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                          (0, 255, 0), thickness)

            # draw label and confidence of prediction in frame resized
            if class_id in CLASSNAMES:
                label = CLASSNAMES[class_id] + ": " + str(confidence)

                per_frame_object.append(CLASSNAMES[class_id])
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontscale, thickness)

                yLeftBottom = max(yLeftBottom, labelSize[1])
                cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                              (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                              (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                            cv2.FONT_HERSHEY_SIMPLEX, fontscale, (0, 0, 0), thickness)
                objects.add(label)
                # Print(label) # print class and confidence


    logger.debug("[INFO] detected object using MbileNetSSD: {}".format(per_frame_object))

    if RULE_OBJECT not in objects:
        logger.debug(f"[ALERT] There are no {RULE_OBJECT} in Frame...")

    return frame


def detect_object(image, method):
    """
    :param image: input image for object detection
    :param method: which method to use yolo or ssd
    :return: return the final image with output
    """


    if method == "SSD":
        out_img = detect_object_ssd(image)

        return out_img

    elif method == "YOLOV3":
        out_img = detect_object_yolo(image)

        return out_img

