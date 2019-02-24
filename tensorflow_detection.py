import os
import numpy as np
import tensorflow as tf 
import requests
import time
from tempfile import TemporaryFile
from urllib.parse import quote
from PIL import Image, ImageDraw
from flask import Flask, request, make_response
import pyrebase

app = Flask(__name__)
firebase_app_ip = os.environ["FIREBASE_SERVICE_SERVICE_HOST"]


class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.i = 0
        self.memo = None
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        boxes, scores, classes, num = self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_np_expanded})

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                    int(boxes[0,i,1]*im_width),
                    int(boxes[0,i,2] * im_height),
                    int(boxes[0,i,3]*im_width))
        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])


    def close(self):
        self.sess.close()
        self.default_graph.close()


model_path = 'faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
odapi = DetectorAPI(path_to_ckpt=model_path)
threshold = 0.4

config = {
    'apiKey': "AIzaSyDhHOVv0cuU7TvydMsoRvvOtv9tDTaS1zk",
    'authDomain': "lavision.firebaseapp.com",
    'databaseURL': "https://lavision.firebaseio.com",
    'projectId': "lavision",
    'storageBucket': "lavision.appspot.com",
     'serviceAccount': 'service-account.json'
}
firebase = pyrebase.initialize_app(config)
storage = firebase.storage()


@app.route("/", methods=['POST'])
def main():
    location_type = request.form["location_type"]
    location_name = request.form["location_name"]
    w, h = 1280, 720
    img = Image.open(request.files['file']).resize((w, h))
    arr = np.array(img)
    arr = arr[:, :, :3]  # R,G,B

    boxes, scores, classes, num = odapi.processFrame(arr)

    # Visualization of the results of a detection.
    num_people = 0
    draw = ImageDraw.Draw(img)
    for i in range(len(boxes)):
        # Class 1 represents human
        if classes[i] == 1 and scores[i] > threshold:
            num_people += 1
            box = list(boxes[i])  # [ymin, xmin, ymax, xmax]
            points = [(box[1],box[0]), (box[3],box[0]), (box[3],box[2]), 
                    (box[1],box[2]), (box[1],box[0])]
            draw.line(points, fill='red', width=2)
    
    img = img.resize((w//2, h//2))
    img.save("current.png")
    storage.child("/current.png").put("current.png")

    requests.post('http://{}/{}/{}'.format(firebase_app_ip,
           quote(location_type), quote(location_name)), 
        data={'num': num_people, 'time': int(time.time())}
    )    
    
    return "OK"


if __name__ == '__main__':
    app.run(host="0.0.0.0")
