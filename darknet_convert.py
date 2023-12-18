import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# path to the labels txt file
labels_file = r"pickle\Images\Labels\pictor_ppe_crowdsourced_approach-01_train.txt"

# directory to save the individual txt files
output_dir = r"pickle\Labels"

# read the labels file
with open(labels_file, "r") as f:
    lines = f.readlines()


# iterate over each line (image) in the labels file
for line in lines:
    # split the line into image name and annotations
    parts = line.strip().split("\t")
    image_name = parts[0]
    annotations = parts[1:]
    yolo_annotations = []
    # create a new txt file for the image
    txt_file = os.path.join(output_dir, os.path.splitext(image_name)[0] + ".txt")
    f = open(txt_file, "w") # create the file
    # iterate over each annotation and write it to the txt file
    for annotation in annotations:
            annotation_parts = annotation.split(',')
            class_num = int(annotation_parts[4])
            x_min = int(annotation_parts[0])
            y_min = int(annotation_parts[1])
            x_max = int(annotation_parts[2])
            y_max = int(annotation_parts[3])

            yolo_annotation = f"{class_num} {x_min},{y_min},{x_max},{y_max}"
            yolo_annotations.append(yolo_annotation)
    f.write(yolo_annotation + "\n")
    f.close()

img = plt.imread(r'image_from_china(1).jpg')

# Create a new figure with a specific size (in inches)
fig, ax = plt.subplots(1, figsize=(12, 9))

ax.imshow(img)


yolo_annotations = [
    '0 818,360,827,368',
    '0 871,366,878,373',
    '0 1000,366,1040,385'
]

for box in yolo_annotations:
    # Split the string by spaces to extract the class_num and coordinates
    class_num, coordinates = box.split()

    # Split the coordinates by comma to extract the individual values
    x_min, y_min, x_max, y_max = map(int, coordinates.split(','))

    # Create a Rectangle patch using the extracted values
    rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

plt.show()

