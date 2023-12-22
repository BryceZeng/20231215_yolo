import os
import shutil
import cv2
import numpy as np



class ImageTransformer:
    def __init__(self, seed=None, n_outputs=1):
        self.seed = seed
        self.n_outputs = n_outputs

    def Noise(self, image_input, label_input, image_output, label_output):
        np.random.seed(self.seed)
        data = list(zip(image_input, label_input))

        for image_file, label_file in data:
            for i in range(self.n_outputs):
                # add noise to the image file
                img = cv2.imread(os.path.join(image_input, image_file)).astype(
                    np.float32
                )
                noise = np.random.normal(0, 5, img.shape)
                noisy_img = img + noise
                # save the noisy image to file
                noisy_filename = os.path.join(
                    image_output, f"{os.path.splitext(image_file)[0]}_noise_{i}.jpg"
                )
                cv2.imwrite(noisy_filename, noisy_img)

                # move the noisy image and the label file
                shutil.move(os.path.join(image_input, image_file), noisy_filename)
                shutil.move(
                    os.path.join(label_input, label_file),
                    os.path.join(label_output, label_file),
                )
        return None

    def Lighting(self, image_input, label_input, image_output, label_output):
        np.random.seed(self.seed)
        data = list(zip(image_input, label_input))

        for image_file, label_file in data:
            for i in range(self.n_outputs):
                # add rand brightness/contrast/saturation to the image file
                img = cv2.imread(os.path.join(image_input, image_file)).astype(
                    np.float32
                )
                # randomly adjust brightness, contrast, and saturation
                brightness = np.random.uniform(-2.0, 2.0)
                contrast = np.random.uniform(-2.0, 2.0)
                saturation = np.random.uniform(-2.0, 2.0)

                img = brightness * img  # apply brightness adjustment
                img = (contrast * (img - 128) + 128).clip(
                    0, 255
                )  # apply contrast adjustment
                img_min, img_max = img.min(), img.max()
                if img_max == img_min:
                    img.fill(128)
                else:
                    img = (img - img_min) / (img_max - img_min)  # normalize the image
                    mean = np.mean(img, axis=(0, 1), keepdims=True)
                    img = mean + saturation * (
                        img - mean
                    )  # apply saturation adjustment
                    img = img * 255.0  # convert back to uint8 format

                # save the modified image to file
                new_filename = os.path.join(
                    image_output, f"{os.path.splitext(image_file)[0]}_lighting_{i}.jpg"
                )
                cv2.imwrite(new_filename, img.astype(np.uint8))

                # move the modified image and the label file
                shutil.move(
                    os.path.join(image_input, image_file),
                    new_filename,
                )
                shutil.move(
                    os.path.join(label_input, label_file),
                    os.path.join(label_output, label_file),
                )
        return None

    def LateralFlip(self, image_input, label_input, image_output, label_output):
        np.random.seed(self.seed)
        data = list(zip(image_input, label_input))

        for image_file, label_file in data:
            for i in range(self.n_outputs):
                img = cv2.imread(os.path.join(image_input, image_file)).astype(
                    np.float32
                )
                # apply lateral flip with 50% probability
                if np.random.rand() < 0.5:
                    img = np.fliplr(img)
                    _, width, _ = img.shape

                    # flip the bounding box coordinates in the label file
                    with open(os.path.join(label_input, label_file)) as f:
                        lines = f.readlines()
                    with open(os.path.join(label_output, label_file), "w") as f:
                        for line in lines:
                            parts = line.strip().split()
                            category, x, y, w, h = (
                                parts[0],
                                float(parts[1]),
                                float(parts[2]),
                                float(parts[3]),
                                float(parts[4]),
                            )
                            x = width - x - w  # flip the x coordinate
                            f.write(f"{category} {x} {y} {w} {h}\n")

                # save the modified image to file
                new_filename = os.path.join(
                    image_output, f"{os.path.splitext(image_file)[0]}_flip_{i}.jpg"
                )
                cv2.imwrite(new_filename, img.astype(np.uint8))

                # move the modified image and the label file
                shutil.move(
                    os.path.join(image_input, image_file),
                    new_filename,
                )
                shutil.move(
                    os.path.join(label_input, label_file),
                    os.path.join(label_output, label_file),
                )
        return None

# ['hue', 'rotation', 'contrast', 'saturation', 'gaussian']




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
                stddev = np.random.randint(8, 16)
                noise = np.zeros(img.shape, np.uint8)
                cv2.randn(noise, mean, stddev)
                img = cv2.add(img, noise)


        output_img_name = os.path.splitext(os.path.basename(img_path))[0]+'_augmented.jpg'
        output_img = os.path.join(output_img_path, output_img_name)
        cv2.imwrite(output_img, img)

        output_label_name = label_name.split('.')[0] + '_augmented.txt'
        output_label = os.path.join(output_label_path, output_label_name)
        with open(output_label, "w") as label_file:
            label_file.write(yolo_label)

