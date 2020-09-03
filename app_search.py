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
import random
from datetime import timedelta,datetime
from flask import Flask, render_template, request, send_from_directory,Blueprint,jsonify

## Global variables
FACE_MODELS_DIR = os.environ.get("FACE_MODELS_DIR", './face_models')
DETECTOR_MODEL_PATH = os.path.join(FACE_MODELS_DIR, 'detector/seeta_fd_frontal_v1.0-c4619d06.bin')
FEATURE_MODEL_PATH = os.path.join(FACE_MODELS_DIR, 'feature/model-20180402-114759')
IMG_TEMP_PATH = './static/images/'

IMAGE_SZ = 160
DIM = 512
DIM_QUERYIMAGE = 1000000 # h*w of downloaded image to avoid long time detecting faces in big images

client = fawn.Fawn('http://172.16.2.180:16000')

## set detectors
detector = Detector(DETECTOR_MODEL_PATH)
detector.set_min_face_size(30)

model = fdv.Model(FEATURE_MODEL_PATH, image_size=IMAGE_SZ)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

print("Load faceNet =>>>")
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
model.loader(sess)
print("Done!\n")

app = Flask(__name__)
app.send_file_max_age_default = timedelta(seconds=1)
FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(filename='fawn_query_api.log',level=logging.INFO, format=FORMAT)

def image_resize(image, dim=1000000, inter = cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    if h * w < dim:
        return image
    else:
        ratio = float(float(h)*float(w) / float(dim))
        size = (int(float(w)/ratio),int(float(h)/ratio)) 
        # resize the image
        resized = cv2.resize(image, size, interpolation = inter)
    return resized

def getnwmlowres(barcode,imagepath,imgurl):
    if '/ids-storage-uk/' in imgurl: # if it's IDS url, get nwmlowres
        try:
            r = requests.get('https://www.idspicturedesk.com/ng/_bfind.jsp?type=nwmlowres&hours=1&barcode='+barcode)
            if len(r.content)<10:
                url = imgurl.replace('/thumbs/','/lowres/')
            else:
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
            nparr = np.frombuffer(r2.content,np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            resized = image_resize(image, dim=DIM_QUERYIMAGE, inter = cv2.INTER_AREA)
            cv2.imwrite(imagepath,resized)
        except Exception as err:
            print('%s %s Image Access Error %s'%(barcode,imagepath,str(err)))
            logging.exception('%s %s Image Access Error %s'%(barcode,imagepath,str(err)))
            return str(err)
    else:
        try:
            resp = requests.get(imgurl,allow_redirects=True).content
            nparr = nparr = np.frombuffer(resp,np.uint8)
            image = cv2.imdecode(neparr, cv2.IMREAD_COLOR)
            resized = image_resize(image, dim=DIM_QUERYIMAGE, inter = cv2.INTER_AREA)
            cv2.imwrite(imagepath,resized)
        except Exception as err:
            print('External Image Access Error, '+str(err))
            logging.exception('%s %s External Image Access Error %s'%(barcode,imagepath,str(err)))
            return str(err)
    return True
    

@app.route('/', methods=['GET'])
def main():
    render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    try:
        threshold = request.json['threshold']
        db = request.json['db']
        url = request.json['url']
        K = request.json['K']
        print('Receive %s %d %s %d'%(url,db,threshold,K))
        logging.info('Receive %s %d %s %d'%(url,db,threshold,K))
    except Exception as err:
        print('Parameter Error: %s'%str(err))
        logging.info('Parameter Error: %s'%str(err))
        return jsonify(error='Parameter Error: %s'%str(err))

    # create imagepath for download
    if '/ids-storage-uk/' in url:
        barcode = url.split('/')[-1].split('.')[0]
    else:
        timestamp = '{:%Y%m%d%H%M%S}'.format(datetime.now())
        barcode = timestamp+'_'+str(random.randint(100,999))
    imagedir = os.path.join(IMG_TEMP_PATH,barcode)
    if os.path.exists(imagedir):
        shutil.rmtree(imagedir)
    os.makedirs(imagedir)
    imagename = barcode+'.jpg'
    imagepath = os.path.join(imagedir,imagename)
    print('downloading image')
    download_result = getnwmlowres(barcode,imagepath,url)
    if download_result != True:
        print(download_result)
        logging.info('Cannot download image %s to disk' % barcode)
        return jsonify(error='Cannot download image %s to disk' % barcode)
    print('finished downloading image')
    # extract the faces
    face_object = fdv.Face(imagepath, imagedir)
    face_object.detect_face(detector=detector)
    face_object.write_clip(write_to_file=True)
    face_object.write_info("info")
    face_object.vectorize(sess=sess, model=model, model_input_size=(IMAGE_SZ, IMAGE_SZ))
    
    # if no face in query image
    if len(face_object.faces) == 0:
        print("No face detected in the query image %s"%url)
        logging.info("No face detected in the query image %s"%url)
        return jsonify(error="No face detected in the query image %s"%url)

    data2return = []
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
        face_position = [round((face.left/face_object.image_size[0]),4),
                       round(float(face.top/face_object.image_size[1]),4),
                       round(float(face.right/face_object.image_size[0]),4),
                       round(float(face.bottom/face_object.image_size[1]),4)]
        # print(face_position)
        data2return.append({'url':url,'facepos':face_position,'hits':[]})

    for i, feature in enumerate(face_object.face_features):
        hits = client.search(feature.reshape(512),K=int(K),db=int(db))
        hits_organised = []
        keys = {}
        for hit in hits:
            if hit['key'].split('_')[0] not in keys:
                keys['key'] = 1
            else:
                continue
            thumb = hit['meta'].split(',',1)[0]
            facepos = [float(item) for item in hit['meta'].split(',',1)[1].split(',')]
            if hit['score'] != 0:
                score_inverse = round( 1/hit['score'],3)
            else:
                score_inverse = 1000000
            if score_inverse < float(threshold):
                continue
            hits_organised.append({'key': hit['key'],'thumbnail':thumb,'score':score_inverse,'facepos':facepos})
        data2return[i]['hits'] = hits_organised
    print(data2return)
    return jsonify(data=data2return)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True) 