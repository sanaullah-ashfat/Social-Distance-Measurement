# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 23:24:51 2020

@author: Md. Sanaullah
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import sqrt
from functions import *
from view import *
import skvideo.io


confidence_threshold = 0.3
nms_threshold = 0.3

min_distance = 115
width = 608
height = 608

config = 'E:/Social-Distancing-AI-master/models/yolov3.cfg'
weights = 'E:/Social-Distancing-AI-master/models/yolov3.weights'
classes = 'E:/Social-Distancing-AI-master/models/coco.names'

with open(classes, 'rt') as f:
    coco_classes = f.read().strip('\n').split('\n')

model = create_model(config, weights)
output_layers = get_output_layers(model)
picture = generate_picture()
video = cv2.VideoCapture('E:/test_y.mp4')
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter('D:/outputbg.avi', fourcc, 20.0, (1250, 800))


while True:
  
    _,frame = video.read()
    
     
    frame =cv2.resize(frame,(1250, 800))
    frame =cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    blob = blob_from_image(frame, (width, height))
    
    outputs = predict(blob, model, output_layers)

    boxes, nms_boxes, class_ids = non_maximum_suppression(frame, outputs, confidence_threshold, nms_threshold)

    person_boxes = get_domain_boxes(coco_classes, class_ids, nms_boxes, boxes, domain_class='person')

    good, bad = people_distances_bird_eye_view(person_boxes, min_distance)

    new_image = draw_new_image_with_boxes(frame, good, bad, min_distance, draw_lines=True)
     
    green_points = [g[6:] for g in good]
    red_points = [r[6:] for r in bad]
    

    
    print("safe",len(green_points))
    print("danger",len(red_points))
    
    bird_eye_view = generate_bird_eye_view(green_points, red_points)
    output_image = generate_content_view(picture, new_image, bird_eye_view)

      
    cv2.imshow("view",output_image)
  
    out.write(new_image)
    key = cv2.waitKey(50)
    if key & 0xFF == ord('q'):
         break
    


video.release()
out.release()
cv2.destroyAllWindows()


