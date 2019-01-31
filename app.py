#!/usr/bin/env python3
import os
import sys
import cv2
import hashlib
import struct

from pyseeta import Detector
import face_detect_vectorize as fdv
import tensorflow as tf
import numpy as np

from datetime import timedelta
from flask import Flask, render_template, request, send_from_directory
import donkey

FACE_MODELS_DIR = os.environ.get("FACE_MODELS_DIR", './face_models')
DETECTOR_MODEL_PATH = os.path.join(FACE_MODELS_DIR, 'detector/seeta_fd_frontal_v1.0-c4619d06.bin')
FEATURE_MODEL_PATH = os.path.join(FACE_MODELS_DIR, 'feature/model-20180402-114759')

# Global variables
IMAGE_SZ = 160
DIM = 512

# feature search engine
client = donkey.Server('donkey.xml', True)

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

# pyseeta detector

app = Flask(__name__)
# cache time
app.send_file_max_age_default = timedelta(seconds=1)

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        imageFile = request.files['file']

        file_bytes = np.asarray(bytearray(imageFile.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        uniqueCode = hashlib.sha1(image).hexdigest()[:16]
        directory = os.path.join('static/images', str(uniqueCode))
        fileUniqueName = str(uniqueCode) + '.jpg'
        os.system("mkdir -p %s" % directory)
        cv2.imwrite(os.path.join(directory, fileUniqueName), image)

        # extract the faces
        face = fdv.Face(os.path.join(directory, fileUniqueName), directory)
        face.detect_face(detector=detector)
        face.write_clip(write_to_file=True)
        face.vectorize(sess=sess, model=model, model_input_size=(IMAGE_SZ, IMAGE_SZ))

        input_faces = [str(i) + ".png" for i in range(face.num_faces)]

        output_faces = {}
        origin_img = {}
        scores = {}
        count = 0

        for feature in face.face_features:
            feature = feature.reshape(DIM)
            query = {'db': 1,
                     'K': 10,
                     'raw': False,
                     'content': struct.pack('%df' % DIM, *feature),
                     'url': '',
                     'type': ''}
            tmp = client.search(query)
            out = []
            org_url = []
            score = []
            for f in tmp['hits']:
                key = f['key']
                score.append(round(f['score'], 3))
                out.append(key + '.png')
                org_url.append(f['meta'])
            scores[input_faces[count]] = score
            output_faces[input_faces[count]] = out
            origin_img[input_faces[count]] = org_url
            count += 1

        columns = ["Similar Face %s" % (i+1) for i in range(10)]

        return render_template('analyze.html', directUniqueName=uniqueCode, fileUniqueName=fileUniqueName,
                               column_names=columns, input_faces=input_faces, output_faces=output_faces,
                               output_scores=scores, origin_img=origin_img)

    return render_template('index.html')

@app.route('/<path:path>')
def send_img(path):
    return send_from_directory('.',path)


if __name__ == '__main__':
    # app.debug = True
    # app

    app.run(host='0.0.0.0', port=8888, debug=True, use_reloader=False)


 
# # @app.route('/upload', methods=['POST', 'GET'])
# @app.route('/upload', methods=['POST', 'GET'])
# def upload():
#     if request.method == 'POST':
#         f = request.files['file']
#
#         if not (f and allowed_file(f.filename)):
#             return jsonify({"error": 1001, "msg": "Please check image format (valid format: png、PNG、jpg、JPG、bmp)"})
#
#         #user_input = request.form.get("name")
#
#         # current path
#         basepath = "/home/ubuntu/face-search-engine"
#
#         # remove existing images
#         filelist = [ file for file in os.listdir(os.path.join(basepath, 'static/images'))]
#         for file in filelist:
#             os.remove(os.path.join(os.path.join(basepath, 'static/images'), file))
#
#         upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))
#         # upload_path = os.path.join(basepath, 'static/images','test.jpg')
#         f.save(upload_path)
#
#         img = cv2.imread(upload_path)
#         cv2.imwrite(os.path.join(basepath, 'static/images', 'test.jpg'), img)
#
#         return render_template('upload_ok.html')#, userinput=user_input)
#
#     return render_template('upload.html')
#
#
# @app.route('/upload/analyze', methods=['POST', 'GET'])
# def analyze():
#     if request.method == 'POST':
#         # current path
#         basepath = os.path.dirname(__file__)
#         print(basepath)
#         path = os.path.join(basepath, 'static/images', 'test.jpg')
#         face = fdv.Face(path, os.path.join(basepath, 'static/images/'))
#         face.detect_face(detector=detector)
#         face.write_clip(write_to_file=True)
#         face.vectorize(sess=sess, model=model, model_input_size=(IMAGE_SZ, IMAGE_SZ))
#
#         input_faces = [str(i)+".png" for i in range(face.num_faces)]
#
#         output_faces = {}
#         origin_img = {}
#         scores = {}
#         count = 0
#         for feature in face.face_features:
#             tmp = client.search(feature.reshape(512), K=5)
#             out = []
#             org_url = []
#             score = []
#             for f in tmp:
#                 key = f['key']
#                 score.append(round(f['score'],5))
#                 out.append(key +'.png')
#                 info = pickle.load(open(os.path.dirname(key) + "/info", "rb"))
#                 org_url.append(info['url'])
#             scores[input_faces[count]] = score
#             output_faces[input_faces[count]] = out
#             origin_img[input_faces[count]] = org_url
#
#             count += 1
#
#         return render_template('analyze.html',input_faces=input_faces, output_faces=output_faces, output_scores=scores, origin_img=origin_img)#, userinput=user_input)
#
#     return render_template('upload.html')
#
# @app.route('/<path:path>')
# def send_img(path):
#     return send_from_directory('/home/ubuntu/face-search-engine/',path)




