import json
import cv2
import glob
import numpy as np
import os
# read json annotation file
directory = r'/home/mkhokhar21/Documents/S3/CV/Homeworks/Project/data/raw/Br35H-Mask-RCNN/TEST_MASK'
with open('/home/mkhokhar21/Documents/S3/CV/Homeworks/Project/data/raw/Br35H-Mask-RCNN/TEST/annotations_test.json') as json_file:
    data = json.load(json_file)
# loading images
images_path = "/home/mkhokhar21/Documents/S3/CV/Homeworks/Project/data/raw/Br35H-Mask-RCNN/TEST/"
TEST_images = glob.glob(images_path + "*.jpg")

for img in TEST_images:
    image = cv2.imread(img)
    dimensions = image.shape
    file_name = str(img).replace("/home/mkhokhar21/Documents/S3/CV/Homeworks/Project/data/raw/Br35H-Mask-RCNN/TEST/\\", "")
    file_name = file_name.replace(".jpg", "")
    os.chdir(directory)
    tmp = np.zeros(dimensions).astype('uint8')
    for d in data:
        path = f"{images_path}{data[d]['filename'].split('.', 1)[0]}"
        if file_name == path:
            if len(data[d]['regions'][0]['shape_attributes']) == 3:
                x_pixels = data[d]['regions'][0]['shape_attributes']['all_points_x']
                y_pixels = data[d]['regions'][0]['shape_attributes']['all_points_y']
                pts = []
                for i in range(len(x_pixels)):
                    pts.append([x_pixels[i], y_pixels[i]])
                ptss = np.array(pts)
                ptss = ptss.reshape((-1, 1, 2))
                isClosed = True
                tmp = cv2.fillPoly(tmp, [ptss], (255,255,255))
            elif len(data[d]['regions'][0]['shape_attributes']) == 6:
                center_coordinates = (data[d]['regions'][0]['shape_attributes']['cx'],
                                        data[d]['regions'][0]['shape_attributes']['cy'])
                axesLength = (int(data[d]['regions'][0]['shape_attributes']['rx']),
                                int(data[d]['regions'][0]['shape_attributes']['ry']))
                angle = data[d]['regions'][0]['shape_attributes']['theta']
                startAngle = 0
                endAngle = 360
                tmp = cv2.ellipse(tmp,
                                    center_coordinates,
                                    axesLength,
                                    angle,
                                    startAngle,
                                    endAngle,
                                    (255,255,255),
                                    thickness=-1)

    cv2.imwrite("{}.png".format(file_name.replace('TEST', 'TEST_MASK')), tmp.astype('uint8'))