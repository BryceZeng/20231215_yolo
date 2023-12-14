import torch

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Define your data configuration
data = dict(
    train = 'datasets/train/images',
    val = 'datasets/valid/images',
    nc = 2,
    names = ['person', 'helmet']
)

# Save data configuration to a YAML file
import yaml
with open('helmet.yaml', 'w') as outfile:
    yaml.dump(data, outfile, default_flow_style=False)

# Train the model
import os
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

# !python yolov5-master/train.py --img 640 --batch 16 --epochs 5 --data helmet.yaml --weights yolov5s.pt --name yolov5s_results

