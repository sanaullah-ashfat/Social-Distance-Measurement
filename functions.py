import numpy as np
import cv2
from math import sqrt

def create_model(config, weights):

    model = cv2.dnn.readNetFromDarknet(config, weights)
    backend = cv2.dnn.DNN_BACKEND_OPENCV
    target = cv2.dnn.DNN_TARGET_CPU
    
    model.setPreferableBackend(backend)
    model.setPreferableTarget(target)
    
    return model

def get_output_layers(model):
 
    layer_names = model.getLayerNames()
    output_layers = [layer_names[i[0]-1] for i in model.getUnconnectedOutLayers()]
    return output_layers

def blob_from_image(image, target_size):

    if not isinstance(target_size, tuple): 
        raise Exception("target_size must be a tuple (width, height)")
    
    blob = cv2.dnn.blobFromImage(image, 
                                 1/255.,
                                 target_size,
                                 [0,0,0],
                                 1,
                                 crop=False)
    
    return blob

def predict(blob, model, output_layers):
   
    model.setInput(blob)
    outputs = model.forward(output_layers)

    return outputs
    
def non_maximum_suppression(image, outputs, confidence_threshold=0.6, nms_threshold=0.4):
    
    class_ids = []
    confidences = []
    boxes = []

    img_height, img_width = image.shape[:2]
    
    #detecting bounding boxing
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                cx = int(detection[0] * img_width)
                cy = int(detection[1] * img_height)
                width = int(detection[2] * img_width)
                height = int(detection[3] * img_height)
                left = int(cx - width / 2)
                top = int(cy - height / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    
    nms_indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    
    return boxes, nms_indices, class_ids

def get_domain_boxes(classes, class_ids, nms_indices, boxes, domain_class):
    

    domain_boxes = []
    for index in nms_indices:
        idx = index[0]
        class_name = classes[class_ids[idx]]
        if class_name == domain_class:
            box = boxes[idx]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            cx = left + int(width / 2)
            cy = top + int(height / 2)
            domain_boxes.append((left, top, width, height, cx, cy))
    
    return domain_boxes

def people_distances_bird_eye_view(boxes, distance_allowed):
    
    people_bad_distances = []
    people_good_distances = []
    # Tomamos los valores center,bottom
    result = __map_points_to_bird_eye_view([[box[4],box[1]+box[3]] for box in boxes])[0]
    # Creamos nuevos bounding boxes con valores mapeados de bird eye view (8 elementos por item)
    # left, top, width, height, cx, cy, bev_cy, bev_cy
    new_boxes = [box + tuple(result) for box, result in zip(boxes, result)]

    for i in range(0, len(new_boxes)-1):
        for j in range(i+1, len(new_boxes)):
            cxi,cyi = new_boxes[i][6:]
            cxj,cyj = new_boxes[j][6:]
            distance = eucledian_distance([cxi,cyi], [cxj,cyj])
            if distance < distance_allowed:
                people_bad_distances.append(new_boxes[i])
                people_bad_distances.append(new_boxes[j])

    people_good_distances = list(set(new_boxes) - set(people_bad_distances))
    people_bad_distances = list(set(people_bad_distances))
    
    return (people_good_distances, people_bad_distances)

def draw_new_image_with_boxes(image, people_good_distances, people_bad_distances, distance_allowed, draw_lines=False):
    
    green = (0, 255, 0)
    red = (0,0,255)
    new_image = image.copy()
    
    
    
    
    
    for person in people_bad_distances:
        left, top, width, height = person[:4]
        cv2.rectangle(new_image, (left, top), (left + width, top + height), red, 2)
    
    for person in people_good_distances:
        left, top, width, height = person[:4]
        cv2.rectangle(new_image, (left, top), (left + width, top + height), green, 2)
    
    
    
    if draw_lines:
        for i in range(0, len(people_bad_distances)-1):
            for j in range(i+1, len(people_bad_distances)):
                cxi,cyi,bevxi,bevyi = people_bad_distances[i][4:]
                cxj,cyj,bevxj,bevyj = people_bad_distances[j][4:]
                distance = eucledian_distance([bevxi, bevyi], [bevxj, bevyj])
                if distance < distance_allowed:
                    xx=eucledian(cxi, cyi, cxj, cyj)
                    xy =str("{:.2f}".format(xx))
                    cv2.putText(new_image,xy+"m",(cxi, cyi-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 40), 2)
                    cv2.line(new_image, (cxi, cyi), (cxj, cyj), red, 2)
                    
    return new_image

def __matrix_bird_eye_view():
    
    return np.array([[ 1.14199333e+00,  6.94076400e+00,  8.88203441e+02],
       [-5.13279159e-01,  7.26783411e+00,  1.02467130e+03],
       [ 9.79674124e-07,  1.99580075e-03,  1.00000000e+00]])

def __map_points_to_bird_eye_view(points):
    
    if not isinstance(points, list):
        raise Exception("poinst must be a list of type [[x1,y1],[x2,y2],...]")
    
    matrix_transformation = __matrix_bird_eye_view()
    new_points = np.array([points], dtype=np.float32)
    
    return cv2.perspectiveTransform(new_points, matrix_transformation)
    
def eucledian_distance(point1, point2):
    
    x1,y1 = point1
    x2,y2 = point2
    return sqrt((x1-x2)**2 + (y1-y2)**2)



def eucledian(cxi, cyi, cxj, cyj):
    
    return sqrt((cxi-cyj)**2 + (cxj-cyj)**2)/3779.527