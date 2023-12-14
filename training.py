import ultralytics
ultralytics.checks()

from ultralytics import YOLO
from IPython.display import display, Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

model = YOLO(f'yolov8n.pt')
results = model.predict(source='https://carbonicsolutions.com/wp-content/uploads/2016/12/Linde-CO2-Truck.jpg', conf=0.25)

# results[0].boxes.xyxy
# results[0].boxes.conf
# results[0].boxes.cls

results = model.train(
    data="helmet2.yaml", epochs=5, batch=1, name="yolov8n_helmet"
)


model2 = YOLO(r'runs\detect\yolov8n_helmet6\weights\best.pt')

image_paths = [r'test\images\000198.jpg',
            r'test\images\000201.jpg',
            r'test\images\000185.jpg']

results2 = model2.predict(image_paths, conf=0.15)

results2[1]

results2 = model2.predict(source=r'test\images\000198.jpg', conf=0.15)
####################################################
####################################################

# Load the image
img = plt.imread(r'test\images\000198.jpg')

# Create a new figure with a specific size (in inches)
fig, ax = plt.subplots(1, figsize=(12, 9))

# Display the image
ax.imshow(img)
# Get the list of bounding boxes
boxes = results2[0].boxes.xyxy

# Loop over the bounding boxes and add them to the plot
for box in boxes:
    # Create a Rectangle patch
    # Note: we need to convert the tensor to numpy array using .numpy() method
    box = box.numpy()
    rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

# Show the plot
plt.show()

img = plt.imread('datasets/train/images/000002.jpg')
plt.figure(figsize=(20, 12))
plt.imshow(img)
plt.show()


# !yolo task=detect \
# mode=predict \
# model=runs/detect/yolov8n_helmet6/weights/best.pt \
# source=datasets/pothole_dataset_v8/valid/images \
# imgsz=1280 \
# name=yolov8n_v8_50e_infer1280 \
# hide_labels=True
# metrics = model.val()

# # Save the trained model
# model.save('trained_helmet.pt')
# torch.save(model.state_dict(), 'trained_helmet.pt')

# # Load the trained model
# model2 = YOLO()  # Initialize the model class.
# model2.load_state_dict(torch.load('trained_helmet.pt'))
# trained_model = YOLO('trained_helmet.pt')

# # Use the trained model to make predictions
# results = trained_model.predict(source='https://media.roboflow.com/notebooks/examples/dog.jpeg', conf=0.25)

# results.model