# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from calendar import EPOCH
import os
import glob
import random
import base64
import cv2
from yolov5 import train
from pascal_voc_writer import Writer
from pathlib import Path
from DlibTracker import DlibTracker
import threading
from datetime import date
import time
from os.path import join, dirname, realpath
from datetime import timedelta, datetime
from flask import Flask, render_template, request, json, session, Response, url_for
import os
import shutil
import random
from flask_migrate import Migrate
from flask_minify import Minify
from sys import exit

from apps.config import config_dict
from apps import create_app, db

from PyQt5.QtGui import QImage
from pascal_voc_io import PascalVocWriter
from pascal_voc_io import PascalVocReader
from yolo_io import YoloReader
from yolo_io import YOLOWriter
import os.path
import sys

# from yolov5.utils.general import labels_to_class_weights


# WARNING: Don't run with debug turned on in production!
DEBUG = (os.getenv('DEBUG', 'False') == 'True')

# The configuration
get_config_mode = 'Debug' if DEBUG else 'Production'

try:

    # Load the configuration using the default values
    app_config = config_dict[get_config_mode.capitalize()]

except KeyError:
    exit('Error: Invalid <config_mode>. Expected values [Debug, Production] ')

app = create_app(app_config)
Migrate(app, db)

if not DEBUG:
    Minify(app=app, html=True, js=False, cssless=False)

if DEBUG:
    app.logger.info('DEBUG            = ' + str(DEBUG))
    app.logger.info('FLASK_ENV        = ' + os.getenv('FLASK_ENV'))
    app.logger.info('Page Compression = ' + 'FALSE' if DEBUG else 'TRUE')
    app.logger.info('DBMS             = ' + app_config.SQLALCHEMY_DATABASE_URI)
    app.logger.info('ASSETS_ROOT      = ' + app_config.ASSETS_ROOT)

##################################################################
# import required libraries

##########################################################################
roi_x = 0
roi_y = 0
roi_w = 0
rou_h = 0
gTracker = None
gLabel = ''
gPath = os.getcwd()
#imgFolderPath = '/home/torquehq/Pictures/AI_Training/video_label-main/hii123'
############################   Camera Streaming  ############################


class VideoCamera():
    def __init__(self, url):
        self.video = cv2.VideoCapture(url)
        self.url = url
        self.error_count = 0

    def __del__(self):
        self.video.release()

    def reset(self):
        self.video.release()
        self.video = cv2.VideoCapture(self.url)
        self.error_count = 0

    def get_frame(self):
        global gTracker, gLabel
        success, image = self.video.read()
        if success:

            if gTracker is not None:

                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                gTracker.update(rgb)
                track_pos = gTracker.getPos()
                x1 = round(track_pos.left())
                x2 = round(track_pos.right())
                y1 = round(track_pos.top())
                y2 = round(track_pos.bottom())
                width = (int(x2) - int(x1))
                height = (int(y2) - int(y1))

                if len(gLabel) > 0:  # save tracking result
                    writer = Writer(gPath + "/" + gLabel + "_" +
                                    str(gTracker.cnt)+".jpg", width, height)
                    writer.addObject(gLabel, int(
                        x1), int(y1), int(x2), int(y2))
                    writer.save(gPath + "/" + gLabel + "_" +
                                str(gTracker.cnt)+".xml")
                    cv2.imwrite(gPath + "/" + gLabel+"_" +
                                str(gTracker.cnt)+'.jpg', image)
                # draw tracking result
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            #ret, jpeg = cv2.imencode('.jpg', cv2.resize(image, (160, 90)))
            ret, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes(), True
        else:
            return None, False


def gen(camera):
    global gTracker
    gTracker = None
    while True:
        try:
            frame, suc = camera.get_frame()
            if suc:
                camera.error_count = 0
            else:
                camera.error_count += 1
                if camera.error_count > 5:
                    camera.reset()
                    return
                elif camera.error_count > 50:
                    ret, jpeg = cv2.imencode('.jpg', cv2.imread(
                        'static/images/no connected.jpg'))
                    frame = jpeg.tobytes()
        except:
            camera.error_count += 1
            if camera.error_count > 5:
                camera.reset()
                return
            elif camera.error_count > 50:
                ret, jpeg = cv2.imencode('.jpg', cv2.imread(
                    'static/images/no connected.jpg'))
                frame = jpeg.tobytes()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

############################   Camera Stream API Url  ############################


@app.route('/video_feed')
def fr_video_feed1():
    url = request.args.get('url')
    return Response(gen(VideoCamera(url)), mimetype='multipart/x-mixed-replace; boundary=frame')

#### Set label and tracking region ########


@app.route('/api/addLabel', methods=['POST'])
def api_addLabel():
    print("---Add label---")
    x = request.form.get('x', type=int)
    y = request.form.get('y', type=int)
    w = request.form.get('w', type=int)
    h = request.form.get('h', type=int)
    label = request.form.get('label')
    global gTracker
    gTracker = DlibTracker()
    gTracker.reset(x, y, w, h)
    global gLabel, gPath
    gLabel = label
    gPath = str(os.getcwd())+"/" + gLabel
    print(gPath)
    Path(gPath).mkdir(parents=True, exist_ok=True)
    Path(gPath + "/Images")
    Path(gPath + "/Labels")
    #print('type: {}, img: {}'.format(type, img))

    # with open(gLabel + '/' + 'data.yaml', 'w') as data:
    # 	data.write('TRAIN_DIR_IMAGES:'+' ''../../' + gLabel +'/Images' + '\n' )
    # 	data.write('TRAIN_DIR_IMAGES:'+' ''../../' + gLabel +'/Labels' + '\n' )
    # 	data.write('VALID_DIR_IMAGES:'+' ''../..' + gLabel +'/Images' + '\n' )
    # 	data.write('VALID_DIR_IMAGES:'+' ''../../' + gLabel +'/Labels' + '\n' )
    # 	data.write('\n')
    # 	data.write('CLASSES: ['+"'"+'__background__'+"'"+','+ "'" + gLabel + "'" ']')
    # 	data.write('\n')
    # 	data.write('NC: 1')
    # 	data.write('\n')
    # 	data.write('SAVE_VALID_PREDICTION_IMAGES: True')

    with open(gLabel + '/' + 'classes.txt', 'w') as data1:
        data1.write(gLabel)

    return json.dumps({
        'status': 200,
        'msg': 'ok'
    })


# @app.route('/')
# def main_register():
#     return render_template('test.html')


print(gPath)
imgFolderPath = (str(os.getcwd()))


@app.route('/convert')
def convertYolo():
    print("---Convert---")
    print(gLabel)
    print(imgFolderPath)
    for file in os.listdir(gLabel):

        print(gLabel)
        if file.endswith(".xml"):
            print(gPath)
            print("Convert", file)

            annotation_no_xml = os.path.splitext(file)[0]

            imagePath = os.path.join(
                imgFolderPath + "/" + gLabel, annotation_no_xml + ".jpg")

            print("Image path:", imagePath)

            image = QImage()
            image.load(imagePath)
            imageShape = [image.height(), image.width(),
                          1 if image.isGrayscale() else 3]
            imgFolderName = os.path.basename(imgFolderPath + "/" + gLabel)
            imgFileName = os.path.basename(imagePath)

            writer = YOLOWriter(imgFolderName, imgFileName,
                                imageShape, localImgPath=imagePath)

            # Read classes.txt
            classListPath = imgFolderPath + "/" + gLabel + "/" + "classes.txt"
            classesFile = open(classListPath, 'r')
            classes = classesFile.read().strip('\n').split('\n')
            classesFile.close()

            # Read VOC file
            filePath = imgFolderPath + "/" + gLabel + "/" + file
            tVocParseReader = PascalVocReader(filePath)
            shapes = tVocParseReader.getShapes()
            num_of_box = len(shapes)

            for i in range(num_of_box):
                label = classes.index(shapes[i][0])
                xmin = shapes[i][1][0][0]
                ymin = shapes[i][1][0][1]
                x_max = shapes[i][1][2][0]
                y_max = shapes[i][1][2][1]

                writer.addBndBox(xmin, ymin, x_max, y_max, label, 0)

            writer.save(targetFile=imgFolderPath + "/" +
                        gLabel + "/" + annotation_no_xml + ".txt")
    return "Nothing"


dataset_path = (str(os.getcwd()))
percentage_test = 20
p = percentage_test/100


@app.route('/SomeFunction')
def SomeFunction():
    global dataset_path
    dataset_path = (str(os.getcwd()) + "/" + gLabel)

    print("Spliting", dataset_path)
    Path(dataset_path).mkdir(parents=True, exist_ok=True)
    Path(dataset_path + "/data").mkdir(parents=True, exist_ok=True)
    Path(dataset_path + "/data/images").mkdir(parents=True, exist_ok=True)
    Path(dataset_path + "/data/labels").mkdir(parents=True, exist_ok=True)
    Path(dataset_path + "/data/images/train").mkdir(parents=True, exist_ok=True)
    Path(dataset_path + "/data/images/valid").mkdir(parents=True, exist_ok=True)
    Path(dataset_path + "/data/labels/train").mkdir(parents=True, exist_ok=True)
    Path(dataset_path + "/data/labels/valid").mkdir(parents=True, exist_ok=True)
    with open(dataset_path + "/data/" + 'data.yaml', 'w') as data:
        data.write('train:'+' '+dataset_path + "/data/images/train" + '\n')
        data.write('val:'+' '+dataset_path + "/data/images/valid" + '\n')
        data.write('\n')
        data.write('nc: 1')
        data.write('\n')

        data.write('names: [' + "'" + gLabel + "'" ']')
        data.write('\n')
        # data.write('NC: 1')
        # data.write('\n')
        data.write('SAVE_VALID_PREDICTION_IMAGES: True')

    print("Spl", dataset_path)
    for pathAndFilename in glob.iglob(os.path.join(dataset_path, "*.jpg")):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        #print("for", gLabel)
        if random.random() <= p:
            print("for", gPath)
            os.system(
                f"cp {dataset_path}/{title}.jpg {gLabel}/data/images/train")
            os.system(
                f"cp {dataset_path}/{title}.txt {gLabel}/data/labels/train")
        else:
            os.system(
                f"cp {dataset_path}/{title}.jpg {gLabel}/data/images/valid")
            os.system(
                f"cp {dataset_path}/{title}.txt {gLabel}/data/labels/valid")
    return "Nothing"


############################   web pages   ############################
# @app.route('/')
# def main_register():
# 	return render_template('test.html')

@app.route('/training')
def training():
    label = request.form.get('label')
    return train.run(data=dataset_path + "/data/" +'data.yaml', weights='yolov5m.pt',name = gLabel)


@app.route('/')
def main_register():
	return render_template('test.html')

#################################################################


if __name__ == "__main__":
    app.run()
