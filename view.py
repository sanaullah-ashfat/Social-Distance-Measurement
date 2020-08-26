import numpy as np
import cv2

def __generate_partial_image(picture, partial_image, position):
  
    if not isinstance(position, tuple):
        raise Exception("position must be a tuple representing x,y coordinates")
    
    
    image_height, image_width = partial_image.shape[:2]
    x, y = position
    picture[x: x + image_height, y: y + image_width] = partial_image

def __generate_text(image, text, target_size, font_scale, color, thickness):
    
    cv2.putText(
        image,
        text,
        target_size,
        fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
        fontScale=font_scale,
        color=color,
        thickness=thickness
    )

def __generate_logo(path_image, target_size=(280,100)):
    
    img_logo = cv2.cvtColor(cv2.imread(path_image), cv2.COLOR_BGR2RGB)
    img_logo = cv2.resize(img_logo, target_size)
    return img_logo

def generate_bird_eye_view(good, bad):
   
    red = (0,0,255)
    green = (0,255,0)
    target_size = (500, 500)

    # Background size
    background = np.zeros((3500, 3500, 3), dtype=np.uint8)

    # Points that respect the distance
    for point in good:
        cv2.circle(background, tuple(point), 25, green, -1)
    
    # Points that don't respect the distance
    for point in bad:
        cv2.circle(background, tuple(point), 25, red, -1)


    # ROI of bird eye view
    cut_posx_min, cut_posx_max = (2000, 3400)
    cut_posy_min, cut_posy_max = ( 200, 2800)

    bird_eye_view = background[cut_posy_min:cut_posy_max, 
                                cut_posx_min:cut_posx_max, 
                                :]

    # Bird Eye View resize
    bird_eye_view_resize = cv2.resize(bird_eye_view, target_size)

    return bird_eye_view_resize

def generate_picture():
    
    text_color = (38, 82, 133)
    target_size = (1250, 1250, 3)
    background = np.ones(target_size, dtype=np.uint8) * 150
    background[0:120,:] = 255
    background[1200:,:] = 255

 


    picture = cv2.copyMakeBorder(background,2,2,2,2, cv2.BORDER_CONSTANT)
    return picture

def generate_content_view(picture, image, bird_eye_view):
   
    content = picture.copy()
     pts = np.array([[430,170],  
                     [700, 100],
                     [1250, 190],
                     [1150,380]], 
                    np.int32)
     cv2.polylines(image, [pts], True, (0,255,230), thickness=2) 

    __generate_partial_image(content, image, position=(0,0))

    __generate_partial_image(content, bird_eye_view, position=(0,0))

    return content