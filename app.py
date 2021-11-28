"""
@author: devdatta supnekar
@created_on: 11/08/2020
@last_updated:
@updated_by 

"""
# imports 
from flask import Flask, Response, redirect, url_for, render_template, request
from config import *
import numpy as np
from object_detection import *
import imutils
import time
import cv2
import os
import datetime
from werkzeug.utils import secure_filename
from log import log_setup
logger = log_setup("main", LOGFILE)


print("[INFO] necessary packages imported...")

# initialize all golbal variables 
outputFrame = None
imageFrame = None
cap = None
frame_toskip = None
frame_number = None
fps = None
duration = None


# function to convert time into total seconds
def get_sec(time_str):
    """
    Get Seconds from time.
    """
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'


############################## FILES UPLOAD #################################

# route for home page
@app.route("/")
def index():
    return render_template("index.html")


# route to upload files
@app.route("/upload", methods=['GET', 'POST'])
def upload():
    # if img is present in files
    if "img" in request.files:
        logger.debug("[INFO] Request received for `{}`...".format("image"))
        file = request.files["img"]
        img_name = secure_filename(file.filename)
        img = os.path.join(IMAGE_UPLOADS, img_name)
        file.save(img)
        logger.debug("[INFO] Image file uploaded succsesfully : - {}".format(img_name))

        timeStamp = time.strftime('%d-%m-%Y_%H-%M-%S')
        logger.debug("[INFO] processing image file...")
        image = cv2.imread(img)

        # detect Objects in the frame
        global imageFrame

        # draw the final bounding box 
        imageFrame = detect_object(image, METHOD)

        logger.debug("[INFO] processing image file completed, sending responce to html page")

        # save the output as temp.jpg
        savepath = os.path.join(OUTPUT_IMAGES, "output_{}_{}".format(timeStamp, ".jpg"))
        cv2.imwrite(savepath, imageFrame)

        return render_template("image.html")


    elif "vid" in request.files:

        file = request.files["vid"]
        vid_file = secure_filename(file.filename)
        logger.debug("[INFO] Request received for `{}`...".format("Video"))
        vid = os.path.join(VIDEO_UPLOADS, vid_file)
        file.save(vid)
        logger.debug("[INFO] Video file uploaded successfully : - {}".format(vid_file))

        global cap, fps, duration
        cap = cv2.VideoCapture(vid)

        # get video FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # total seconds in video
        total_seconds = frameCount // fps

        # video duration
        duration = str(datetime.timedelta(seconds=total_seconds))

        # print info on server side
        logger.debug("[INFO] fps : {} | Total Frames : {}".format(int(fps), frameCount))
        logger.debug("[INFO] Video Duration: {}".format(duration))
        logger.debug("[INFO] processing video...")

        return redirect(url_for("video", time=duration))


########################### IMAGE PROCESSING ####################################


# function to encode image files and send to client
def GetImage():
    global imageFrame

    # encode the frame in JPEG format
    (flag, img) = cv2.imencode(".jpg", imageFrame)

    while True:
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(img) + b'\r\n')


# route to display image output
@app.route("/display_image")
def display_image():
    return Response(GetImage(), mimetype="multipart/x-mixed-replace; boundary=frame")


############################# VIDEO PROCESSING ##################################

# display video endpoints
@app.route("/video")
def video():
    global duration
    # return the rendered template
    return render_template("video.html", time=duration)


# function to encode video frames
def GetVideo(frame_number=0, frame_skip=1):
    # grab global references to the output frame and lock variables
    global cap, outputFrame

    # jump to spcific frame
    cap.set(1, frame_number)

    # counter for frames
    count = 0

    # loop over frames from the output stream
    while True:

        ret, frame = cap.read()

        if ret:

            # process only Nth frame 
            if count % frame_skip == 0:

                # resize the frame 
                frame = imutils.resize(frame, width=600)

                # DEBUG PRING
                # print("[INFO] frame no: {}".format(count) )

                # detect the emotion and return the final frame with bounding box 
                final_frame = detect_object(frame, METHOD)

                # acquire the lock, set the output frame, and release the
                # lock
                outputFrame = final_frame.copy()

                # check if the output frame is available, otherwise skip
                # the iteration of the loop
                if outputFrame is None:
                    continue

                # encode the frame in JPEG format
                (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

                # ensure the frame was successfully encoded
                if not flag:
                    continue

                # yield the output frame in the byte format
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                       bytearray(encodedImage) + b'\r\n')

            count += 1

        else:
            break


# stream video to endpoints
@app.route("/video_stream")
def video_stream():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(GetVideo(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


########################## FRAME SKIPING IN VIDEO ##################################

# accecpt user input and skiped video 
@app.route("/skip_frames")
def skip_frames():
    # set global variables
    global frame_toskip, frame_number, fps, duration

    # get duration and frames to skip 
    input_duration = request.args.get('time')
    input_frame = request.args.get('frameskip')

    # check if the are empty or not and enter the defualt values
    if input_duration == "" and input_frame != "":
        input_duration = "00:00:00"
    elif input_duration != "" and input_frame == "":
        input_frame = 1
    elif input_duration == "" and input_frame == "":
        input_duration = "00:00:00"
        input_frame = 1

    # print the info on server side 
    logger.debug("[INFO] Skiping video at : {}".format(input_duration))
    logger.debug("[INFO] No of Frames to Skip : {}".format(input_frame))

    # calculate total sececonds 
    t_secs = get_sec(input_duration)

    # cal frame_no based on sec
    frame_number = t_secs * fps
    frame_toskip = int(input_frame)

    # render new video templates
    return render_template('video_skip.html', time=duration)


# stream the skipped video files to endpoints
@app.route("/video_skip")
def video_skip():
    # set global variables
    global frame_toskip, frame_number

    # return the response generated along with the specific media
    # type (mime type)
    return Response(GetVideo(frame_number, frame_toskip),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host=SERVER_URL, port=SERVER_PORT, debug=True,
            threaded=True, use_reloader=True)
