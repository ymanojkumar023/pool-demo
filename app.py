from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

import time
from math import ceil

# Keras

from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_L2Normalization import L2Normalization
from keras.models import load_model
from imageio import imread
from keras.preprocessing import image
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
from matplotlib import pyplot as plt
#plt.box(False)

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# print('Setting parameters and loading our trained Model... ')
img_height = 300 # Height of the model input images
img_width = 300 # Width of the model input images
img_channels = 3 # Number of color channels of the model input images
mean_color = [123, 117, 104] # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
n_classes = 1 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
scales = scales_pascal
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100, 300] # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are divided as in the original implementation
normalize_coords = True

__file__  = 'uploads'


# Define a flask app
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Model saved with Keras model.save()
# MODEL_PATH = 'models/Pool_det_model_epoch-20_loss-3.5767_val_loss-3.9496.h5'

# Load our trained model
# We need to create an SSDLoss object in order to pass that to the model loader.
#ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
#model = load_model(MODEL_PATH, custom_objects={'AnchorBoxes': AnchorBoxes,
#                                               'L2Normalization': L2Normalization,
#                                               'compute_loss': ssd_loss.compute_loss})                                               

#model._make_predict_function()          # Necessary

print('Model loaded. Start serving...')

print('Model loaded. Check http://127.0.0.1:5001/')


"""
def model_predict(img_path, model):
    input_images = [] # Store resized versions of the images here.
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img = image.img_to_array(img)
    input_images.append(img)
    input_images = np.array(input_images) 
    preds = model.predict(input_images) #Prediction using trained model   
    return preds
"""

def save_prediction(img_path, pred_decoded):
    orig_image = imread(img_path)
    
    colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist()
    classes = ['background','pool']
    plt.imshow(orig_image)
    current_axis = plt.gca()
    plt.axis('off')
    plt.savefig('static/preds/predicted_img.jpg', bbox_inches='tight', pad_inches = 0)
    plt.clf()
    plt.cla()
    plt.close()
'''    for box in pred_decoded[0]:
        # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
        xmin = box[-4] * orig_image.shape[1] / img_width
        ymin = box[-3] * orig_image.shape[0] / img_height
        xmax = box[-2] * orig_image.shape[1] / img_width
        ymax = box[-1] * orig_image.shape[0] / img_height
        color = colors[int(box[0])]
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
        current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})
    plt.axis('off')
    plt.savefig('static/preds/predicted_img.jpg', bbox_inches='tight', pad_inches = 0)
    plt.clf()
    plt.cla()
    plt.close()
'''

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path) 
        
        
        result = str('No Pool detected in the image.')
        # Call fucntion to save the predicted image with boundary boxes drawn
        try: os.remove('static/preds/predicted_img.jpg')
        except: None
        save_prediction(file_path, None)
        
        # Delete the uploaded image from storage
        os.remove(file_path)
        
        return result
    return None

'''        # Make prediction
        preds = model_predict(file_path, model)
        y_pred_decoded = decode_detections(preds, confidence_thresh=0.5, iou_threshold=0.4, top_k=200,
                                       normalize_coords=normalize_coords, img_height=img_height, img_width=img_width)
        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        
        var2 = ''
        pred_len = len(y_pred_decoded[0])
        if pred_len > 0:
            # Predicted text message
            for i in range(0, pred_len):
                var2 = var2+ str(round(y_pred_decoded[0][i][1],2)) +str(', ')
                result = str(pred_len)+str(' Pool/s detected in the image; Confidence: ')+str(var2)
                result = result.rstrip().rstrip(',')
        else:
            result = str('No Pool detected in the image.')
            
        # Call fucntion to save the predicted image with boundary boxes drawn
        try: os.remove('static/preds/predicted_img.jpg')
        except: None
        save_prediction(file_path, y_pred_decoded)
        
        # Delete the uploaded image from storage
        os.remove(file_path)
        
        return result
    return None
'''    

if __name__ == '__main__':
    # app.run(port=5001, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
    
