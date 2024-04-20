
import os
import subprocess
import yaml

def run_command(command):
    """
    Purpose: Execute a shell command
    Input: command (str) - The command to execute
    """
    # Here we use the run() method to execute the command and capture the result
    result = subprocess.run(command, shell=True)
    # If the command failed, print an error message
    if result.returncode != 0:
        print(f"Error executing command: {' '.join(command)}")
    else:
        # otherwise, print a success message
        print(f"Command executed successfully: {' '.join(command)}")

# Here we use the run_command() function to execute the shell command to clone the YOLOv5 repository
run_command('git clone https://github.com/ultralytics/yolov5')

# We change into the yolov5 directory
os.chdir('yolov5')

# We install all required dependencies from the requirements.txt file within the yolov5 directory
run_command('pip install -r requirements.txt')

# Here we create a YAML file that contains all the pathways to the training, validation, and testing datasets
# We also define the number of classes and the names of the classes
dataset_yaml = {
    'path': '/content/drive/MyDrive/DL_Individual_Project_Engandered_Species/YOLO_Data',
    'train': 'images/train',
    'val': 'images/val',
    'test': 'images/test',
    'nc': 29,
    'names': [
        'antelope', 'bat', 'bear', 'butterfly', 'cat',
        'chimpanzee', 'coyote', 'dolphin', 'eagle',
        'elephant', 'gorilla', 'hippopotamus', 'rhinoceros',
        'hummingbird', 'kangaroo', 'koala', 'leopard', 'lion',
        'lizard', 'orangutan', 'panda', 'penguin', 'seal',
        'shark', 'tiger', 'turtle', 'whale', 'zebra', 'bee'
    ]
}
# Here we write the dataset_yaml dictionary to a YAML file
with open('/content/drive/MyDrive/CV_Project_2/Data/YOLO/dataset.yaml', 'w') as file:
    yaml.dump(dataset_yaml, file, default_flow_style=False)

# Here we create a YAML file that contains the hyperparameters for the YOLOv5 model
# We modify the hyperparameters to be more suitable for our dataset
existing_hyp_path = '/content/yolov5/data/hyps/hyp.scratch-low.yaml'
with open(existing_hyp_path, 'r') as file:
    hyp = yaml.safe_load(file)

hyp['optimizer'] = 'adam'
hyp['mosaic'] = 1.0
hyp['mixup'] = 0.2
hyp['jitter'] = 0.2
hyp['hsv_h'] = 0.015
hyp['hsv_s'] = 0.7
hyp['hsv_v'] = 0.4
hyp['degrees'] = 0.2
hyp['translate'] = 0.1
hyp['scale'] = 0.5
hyp['shear'] = 0.1

# Here we write the hyp dictionary to a YAML file
custom_hyp_path = '/content/yolov5/data/hyps/hyp.custom.yaml'
with open(custom_hyp_path, 'w') as file:
    yaml.dump(hyp, file)

# Here we create a directory to store the results of the training and validation runs
project_dir = '/content/drive/MyDrive/DL_Individual_Project_Engandered_Species/YOLO_Notebooks/Runs/Validation/'
os.makedirs(project_dir, exist_ok=True)

# Train the model
run_command(f"python train.py --img 640 --batch 16 --epochs 300 \
--data '/content/drive/MyDrive/CV_Project_2/Data/YOLO/dataset.yaml' \
--hyp '/content/yolov5/data/hyps/hyp.custom.yaml' \
--weights yolov5s.pt \
--cache \
--project '/content/drive/MyDrive/DL_Individual_Project_Engandered_Species/YOLO_Notebooks/Runs/YOLO_Training' \
--name Final_YOLOv5_Endangered_Species_Main")

# Perform inference
run_command(f"python val.py \
--weights '/content/drive/MyDrive/DL_Individual_Project_Engandered_Species/YOLO_Notebooks/Runs/YOLO_Training/Final_YOLOv5_Endangered_Species_Main/weights/best.pt' \
--data '/content/drive/MyDrive/CV_Project_2/Data/YOLO/dataset.yaml' \
--img 640 \
--batch 32 \
--task test \
--project '/content/drive/MyDrive/DL_Individual_Project_Engandered_Species/YOLO_Notebooks/Runs/Test/' \
--name Final_YOLOv5_Endangered_Species_Frozen_10_Test \
--exist-ok")
