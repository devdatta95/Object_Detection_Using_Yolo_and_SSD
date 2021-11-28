# Realtime Obejct Recognition Using YOLO and SSD

Realtime object recognition using the OpenCV 3.3 dnn module + pretrained MobileNetSSD caffemodel.

## Installation
All the dependencies can be installed using `pip`. Just use the following command from the root directory of the project.
```bash
pip3 install -r requirements.txt
```

## How to run this script?
There are two options for video source:

 * Webcam
 * Android device running IP Camera (https://play.google.com/store/apps/details?id=com.pas.webcam&hl=en)

To run the script using webcam as source :

```bash
python3 real_time_object_detection.py  --source webcam
```

To run the script using IP Webcam as source, open the `real_time_object_detection.py` and edit the following line to match your host :

```python
host = 'http://192.168.0.101:8080/'
```

Then to run the script using IP as source :

```bash
python3 real_time_object_detection.py  --source web
```

For any questions, create an issue in this repository.
