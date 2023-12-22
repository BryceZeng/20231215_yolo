import os
import xml.etree.ElementTree as ET
def convert_xmls_to_txt(input_dir, output_dir, label_map=None):
    # create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # loop over all XML files in input directory
    for file in os.listdir(input_dir):
        if file.endswith(".xml"):
            # parse XML file and extract bounding box info
            xml_path = os.path.join(input_dir, file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find("size")
            width = int(size.find("width").text)
            height = int(size.find("height").text)
            objects = root.findall("object")
            bboxes = []
            for obj in objects:
                bbox = obj.find("bndbox")
                xmin = int(bbox.find("xmin").text)
                ymin = int(bbox.find("ymin").text)
                xmax = int(bbox.find("xmax").text)
                ymax = int(bbox.find("ymax").text)
                label = obj.find("name").text
                if label_map is not None and label in label_map:
                    label = label_map[label]
                else:
                    continue
                x_center = ((xmin + xmax) / 2) / width
                y_center = ((ymin + ymax) / 2) / height
                bbox_width = (xmax - xmin) / width
                bbox_height = (ymax - ymin) / height
                bboxes.append([label, x_center, y_center, bbox_width, bbox_height])

            # write bounding box info to text file
            txt_file = os.path.splitext(file)[0] + ".txt"
            txt_path = os.path.join(output_dir, txt_file)
            with open(txt_path, "w") as f:
                for bbox in bboxes:
                    bbox_str = " ".join([str(item) for item in bbox])
                    f.write(bbox_str + "\n")



input_dir = "VOC2028/Annotations"
output_dir = "VOC2028/datasets"
label_map = {
    "person": 0,
    "hat": 1,
}
convert_xmls_to_txt(input_dir, output_dir, label_map)
####
input_dir = r"linde_helmet\labels"
output_dir = r"linde_helmet\labels"
label_map = {
    "Head_with_Helmet": 1,
    "Head_without_covering": 0,
}
convert_xmls_to_txt(input_dir, output_dir, label_map)
######################################
import os
import shutil
from sklearn.model_selection import train_test_split
image_dir = 'VOC2028/JPEGImages'
label_dir = 'VOC2028/datasets'

image_files = os.listdir(image_dir)
label_files = os.listdir(label_dir)

data = list(zip(image_files, label_files))

train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

os.makedirs('datasets/train/images', exist_ok=True)
os.makedirs('datasets/train/labels', exist_ok=True)
os.makedirs('datasets/test/images', exist_ok=True)
os.makedirs('datasets/test/labels', exist_ok=True)

for image_file, label_file in train_data:
    shutil.move(os.path.join(image_dir, image_file), os.path.join('datasets/train/images', image_file))
    shutil.move(os.path.join(label_dir, label_file), os.path.join('datasets/train/labels', label_file))

for image_file, label_file in test_data:
    shutil.move(os.path.join(image_dir, image_file), os.path.join('datasets/test/images', image_file))
    shutil.move(os.path.join(label_dir, label_file), os.path.join('datasets/test/labels', label_file))

import os

def add_prefix_to_files(directory, prefix):
    for filename in os.listdir(directory):
        if not filename.startswith(prefix):  # Avoid renaming already renamed files
            new_filename = prefix + filename
            source = os.path.join(directory, filename)
            destination = os.path.join(directory, new_filename)
            os.rename(source, destination)  # Rename the file

# Use the function
add_prefix_to_files('datasets/val/labels', 'linde_')
