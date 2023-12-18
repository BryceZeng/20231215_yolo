import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def yolo_format(label_path, width, height):
    with open(label_path, "r") as label_file:
        label_data = label_file.read().splitlines()
        yolo_label = ""
        for det in label_data:
            det_data = det.split(" ")
            class_num = int(float(det_data[0]))
            center_x = float(det_data[1]) * width
            center_y = float(det_data[2]) * height
            obj_width = float(det_data[3]) * width
            obj_height = float(det_data[4]) * height
            x_min = int(center_x - obj_width/2)
            y_min = int(center_y - obj_height/2)
            x_max = int(center_x + obj_width/2)
            y_max = int(center_y + obj_height/2)
            yolo_det = f"{class_num} {x_min},{y_min},{x_max},{y_max}\n"
            yolo_label += yolo_det
    return yolo_label


yolo_format(r'datasets\val\labels\linde_000006.txt',400,500)
input_img_path = r'datasets\val\images'
input_label_path = r'datasets\val\labels'

def augment_data(input_img_path, input_label_path, output_img_path, output_label_path, augment_list):
    img_list = os.listdir(input_img_path)
    if not os.path.exists(output_img_path):
        os.makedirs(output_img_path)
    if not os.path.exists(output_label_path):
        os.makedirs(output_label_path)
    for img in img_list:
        img_path = os.path.join(input_img_path, img)
        img = cv2.imread(img_path)
        height, width = img.shape[:2]



    label_name = os.path.splitext(os.path.basename(img_path))[0]+'.txt'
    label_path = os.path.join(input_label_path, label_name)
    yolo_label = yolo_format(label_path, width, height)

    for augment in augment_list:
        if augment == 'hue':
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            random_hue = np.random.randint(-25, 25)
            img_hsv[:, :, 0] = (img_hsv[:, :, 0] + random_hue) % 180
            img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

        elif augment == 'rotation':
            angle = np.random.randint(-10, 10)
            rot_mat = cv2.getRotationMatrix2D((width//2, height//2), angle, 1.0)
            img = cv2.warpAffine(img, rot_mat, (width, height))
            plt.imshow(img)

        elif augment == 'contrast':
            alpha = np.random.uniform(0.8, 1.2)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)

        elif augment == 'saturation':
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            random_sat = np.random.randint(-25, 25)
            img_hsv[:, :, 1] = cv2.add(img_hsv[:, :, 1], random_sat)
            img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

        elif augment == 'gaussian':
            mean = 0
            stddev = np.random.randint(8, 160)
            noise = np.zeros(img.shape, np.uint8)
            noise = cv2.randn(noise, mean, stddev)
            img = cv2.add(img, noise)


    output_img_name = os.path.splitext(os.path.basename(img_path))[0]+'_augmented.jpg'
    output_img = os.path.join(output_img_path, output_img_name)
    cv2.imwrite(output_img, img)

    output_label_name = label_name.split('.')[0] + '_augmented.txt'
    output_label = os.path.join(output_label_path, output_label_name)
    with open(output_label, "w") as label_file:
        label_file.write(yolo_label)