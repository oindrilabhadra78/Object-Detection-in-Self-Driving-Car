# -*- coding: utf-8 -*-
"""
Created on Tue March 10 00:36:18 2020

@author: Oindrila
"""


# import packages
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import tkinter
import cv2
import pathlib

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# print (tf.__version__)
# print (cv2.__version__)

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

import utils_ops
import label_map_util
import vis_util

print ("Successfully imported modules")

# Download mobilenet model for object detection
MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'

# Download from url
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

print ("Successfully downloaded base from URL")

# List of strings that is used to add label to each box.
# Change this url to where your mscoco_label_map.pbtxt file is stored
PATH_TO_LABELS = 'D:/4thSemester/CS299/InnovationLabProject/object_detection/mscoco_label_map.pbtxt'

# Un-tar each tar.gz file
tar_file = tarfile.open(MODEL_FILE)
# print (tar_file.getmembers())
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  
  # Use the frozen inference graph
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())
    
    
print ("Successfully un-tared tar.gz file")
    
detection_graph = tf.Graph()

with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
# print (category_index)

print ("Successfully created category index")

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
          
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
        
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
        
  return output_dict

print ('Start capturing images')

c=0
dir = 'frames_with_objects'
if not os.path.isdir(dir):
    try:
        # creating a folder named data        
        os.makedirs(dir)

        # if not created then raise error
    except OSError:
        print('Error creating directory')

# For webcam. Change the value for external webcam. 
# cap = cv2.VideoCapture(0)

# For IP Camera
# cap = cv2.VideoCapture('rtsp://admin:abc123@192.168.43.89/live/ch00_0')

# For IPWebcam app https://play.google.com/store/apps/details?id=com.pas.webcam
# cap = cv2.VideoCapture('http://192.168.137.253:8080/video')

# For test video
cap = cv2.VideoCapture('test.mp4')

ret = True
while (ret):
    ret, image_np = cap.read()
    if not ret:
        break
    c=c+1
    print ("Frame number"+str(c))
    image_np_expanded = np.expand_dims(image_np, axis=0)
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)
    
    img_path = os.path.join('./'+dir+'/','img' + str(c) + '.jpg')
    plt.savefig(img_path) 
    
    if cv2.waitKey(25) & 0xff == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

# For testing with single images
 
# Add full path to test image directory
# PATH_TO_TEST_IMAGES_DIR = pathlib.Path('')

# Replace 2nd argument with names of image files
# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# for image_path in TEST_IMAGE_PATHS:
#   image = Image.open(image_path)
#   # the array based representation of the image will be used later in order to prepare the
#   # result image with boxes and labels on it.
#   image_np = load_image_into_numpy_array(image)
#   # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
#   image_np_expanded = np.expand_dims(image_np, axis=0)
#   # Actual detection.
#   output_dict = run_inference_for_single_image(image_np, detection_graph)
#   # Visualization of the results of a detection.
#   vis_util.visualize_boxes_and_labels_on_image_array(
#       image_np,
#       output_dict['detection_boxes'],
#       output_dict['detection_classes'],
#       output_dict['detection_scores'],
#       category_index,
#       instance_masks=output_dict.get('detection_masks'),
#       use_normalized_coordinates=True,
#       line_thickness=8)
#   plt.figure(figsize=IMAGE_SIZE)
#   plt.imshow(image_np)
#   c=c+1
#   plt.savefig(str(c)+"my.jpg")
