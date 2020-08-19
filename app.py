#!/usr/bin/env python3
import os
import sys
import cv2
import hashlib
import struct
import requests
from pyseeta import Detector
import face_detect_vectorize as fdv
import fawn
import tensorflow as tf
import numpy as np
import shutil
import logging

from datetime import timedelta
from flask import Flask, render_template, request, send_from_directory,Blueprint,jsonify
import donkey



## Global variables
FACE_MODELS_DIR = os.environ.get("FACE_MODELS_DIR", './face_models')
DETECTOR_MODEL_PATH = os.path.join(FACE_MODELS_DIR, 'detector/seeta_fd_frontal_v1.0-c4619d06.bin')
FEATURE_MODEL_PATH = os.path.join(FACE_MODELS_DIR, 'feature/model-20180402-114759')
IMG_TEMP_PATH = './static/images/'

IMAGE_SZ = 160
DIM = 512

## Face detector
detector = Detector(DETECTOR_MODEL_PATH)
detector.set_min_face_size(30)
model = fdv.Model(FEATURE_MODEL_PATH, image_size=IMAGE_SZ)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
## load face net
print("Load faceNet =>>>")
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
model.loader(sess)
print("Done!\n")

# fawn server 
# client = fawn.Fawn('http://172.16.2.180:16000')
## Init flask
app = Flask(__name__)
app.send_file_max_age_default = timedelta(seconds=1)

## logging
FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(filename='fawn_poll_faces.log',level=logging.INFO, format=FORMAT)

def getnwmlowres(barcode,imagepath,imgurl):
    if '/ids-storage-uk/' in imgurl: # if it's IDS url, get nwmlowres
        try:
            r = requests.get('https://www.idspicturedesk.com/ng/_bfind.jsp?type=nwmlowres&hours=1&barcode='+barcode)
            url = r.text
            url = url.strip()
            r2 = requests.get(url,allow_redirects=True)
            if r2.status_code == 404:
                r = requests.get('https://www.idspicturedesk.com/ng/_bfind.jsp?barcode='+barcode)
                print('%s Cannot get non-water marked image, get original image'%barcode)
                url = r.text
                url = url.strip()
                r2 = requests.get(url,allow_redirects=True)
        except Exception as err:          
            print('%s Cannot get image'%barcode)
            logging.exception('%s Cannot get image'%barcode)
            return str(err)
        try:
            imgcontent = r2.content
            open(imagepath,'wb').write(imgcontent)
        except Exception as err:
            print('%s %s Image Access Error %s'%(barcode,imagepath,str(err)))
            logging.exception('%s %s Image Access Error %s'%(barcode,imagepath,str(err)))
            return str(err)
    else:
        try:
            resp = requests.get(imgurl,allow_redirects=True).raw
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            cv2.imwrite(imagepath,image)
        except Exception as err:
            print('External Image Access Error, '+str(err))
            logging.exception('%s %s External Image Access Error %s'%(barcodeimagepath,str(err)))
            return str(err)
    return True

## Get barcode return feature
@app.route('/', methods=['GET', 'POST'])
def main():
    try:
        barcode = request.json["barcode"]
        imgurl = request.json['url']
        db = request.json['db']
        fawns = request.json['fawns'] # extract and insert for every fawn server
        print(barcode,imgurl,db,fawns)
    except Exception as err:
        print('Having problems getting parameters from fawn_driver %s'%str(err))
        logging.exception('Having problems getting parameters from fawn_driver %s'%str(err))
        return jsonify(error='Having problems getting parameters from fawn_driver %s'%str(err))

    ## create image dir, download image
    imagedir = os.path.join(IMG_TEMP_PATH,barcode)
    if os.path.exists(imagedir):
        shutil.rmtree(imagedir)
    os.makedirs(imagedir)
    imagename = barcode+'.jpg'
    imagepath = os.path.join(imagedir,imagename)
    if getnwmlowres(barcode,imagepath,imgurl) != True:
        logging.info('Cannot download image %s to disk' % barcode)
        return jsonify(error='Cannot download image to disk')
    
     # extract the faces
    face_object = fdv.Face(imagepath, imagedir)
    face_object.detect_face(detector=detector)
    face_object.write_clip(write_to_file=True)
    face_object.write_info("info")
    face_object.vectorize(sess=sess, model=model, model_input_size=(IMAGE_SZ, IMAGE_SZ))

    data2return = [] # data to return to fawn_driver
    # get face coordinates
    for i, face in enumerate(face_object.faces):
        if face.left < 0:
            face.left = 0
        if face.top < 0:
            face.top = 0
        if face.right > face_object.image_size[1]:
            face.right = face_object.image_size[1]
        if face.bottom > face_object.image_size[0]:
            face.bottom = face_object.image_size[0]
        # print(face.top,face.left,face.right,face.bottom)
        face_position = [round((face.left/face_object.image_size[0]),3),
                       round(float(face.top/face_object.image_size[1]),3),
                       round(float(face.right/face_object.image_size[0]),3),
                       round(float(face.bottom/face_object.image_size[1]),3)]
        # print(face_position)
        data2return.append({'key':barcode+'_'+str(i),'postion':face_position})
    # print(data2return)
    # get face feature and insert to db
    for i, feature in enumerate(face_object.face_features):
        feature_reshape = feature.reshape(DIM)
        meta = imgurl
        for x in data2return[i]['postion']:
            meta += ','
            meta += str(x)
        key = data2return[i]['key']
        #print(db)
        for fawn_url in fawns:
            try:
                client = fawn.Fawn(fawn_url)
                client.insert(db,key,feature_reshape,meta)
                print('inserted %s %s to %s' % (data2return[i]['key'],meta,fawn_url))
                logging.info('inserted %s %s to %s db %d' % (data2return[i]['key'],meta,fawn_url,db))
            except Exception as err:
                error = 'Cannot insert to %s with %s %s'%(fawn_url,key,str(err))
                print(error)
                logging.exception('Cannot insert to %s with %s %s'%(fawn_url,key,str(err))) 
                return jsonify(error=error)
    #print(data2return)
    return jsonify(success="success")

#app.register_blueprint(bp)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True)   
        
        


    
