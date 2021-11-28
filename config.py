# flask ip
SERVER_URL = "0.0.0.0"

# flask port
SERVER_PORT = 8090

LOGFILE = "logs/main.log"

METHOD = "SSD"  # (SSD/YOLOV3)
# upload folders
IMAGE_UPLOADS = "uploads/images"
VIDEO_UPLOADS = "uploads/videos"
OUTPUT_IMAGES = "uploads/output"

ENABLE_GPU = False
CONF = 0.2
MODEL = "yolov3"  # "yolov3-tiny
W_CONF = True

PROTOTXT = "models/MobileNetSSD_deploy.prototxt.txt"
CAFFEMODEL = "models/MobileNetSSD_deploy.caffemodel"

RULE_OBJECT = "tvmonitor"

CLASSNAMES = {0: "background",
              1: "aeroplane",
              2: "bicycle",
              3: "bird",
              4: "boat",
              5: "bottle",
              6: "bus",
              7: "car",
              8: "cat",
              9: "chair",
              10: "cow",
              11: "dinigtable",
              12: "dog",
              13: "horse",
              14: "motorbike",
              15: "person",
              16: "pottedplant",
              17: "sheep",
              18: "sofa",
              19: "train",
              20: "tvmonitor"}
