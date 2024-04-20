
import os
import subprocess
import yaml
from s3_data_storage import setup_and_download_from_s3


def run_command(command):
    """
    Purpose: Execute a shell command. Since alot of the YOLOv5 training involves using the command line, this function
    is used to execute shell commands.
    Input: command (str) - The command to execute
    """
    result = subprocess.run(command, shell=True)
    # If the command failed, print an error message
    if result.returncode != 0:
        print(f"Error executing command: {command}")
    else:
        print(f"Command executed successfully: {command}")

# Here is where the AWS access keys and bucket name are defined
# Please reach out to Sri Veerisetti at sri.veerisetti@duke.edu to gain access to the AWS bucket
aws_access_key_id = 'YOUR_AWS_ACCESS_KEY_ID'
aws_secret_access_key = 'YOUR_AWS_SECRET_ACCESS_KEY'
bucket_name = 'YOUR_BUCKET_NAME'
images_s3_folder = 'images'
labels_s3_folder = 'labels'
# Define the local directory where the images and labels will be downloaded
local_dir = 'YOUR_LOCAL_DIRECTORY'

# We can use the setup_and_download_from_s3 function to download the images and labels from the S3 bucket
# The setup_and_download_from_s3 function is defined in the s3_data_storage.py file
setup_and_download_from_s3(aws_access_key_id, aws_secret_access_key, bucket_name, images_s3_folder, os.path.join(local_dir, 'images'))
setup_and_download_from_s3(aws_access_key_id, aws_secret_access_key, bucket_name, labels_s3_folder, os.path.join(local_dir, 'labels'))

# To prepare the YOLOv5 model for training, we need to clone the YOLOv5 repository from GitHub
os.chdir(os.path.join(local_dir, 'workspace')) 
# We can use the run_command function to execute the shell command to clone the YOLOv5 repository 
run_command('git clone https://github.com/ultralytics/yolov5')
# We need to change into the yolov5 directory to install the required dependencies
os.chdir('yolov5')
# Get the requirements for the YOLOv5 model
run_command('pip install -r requirements.txt')

# Here is where the YAML file that contains the pathways to the training, validation, and testing datasets is created
# The classes are defined in the 'names' field
dataset_yaml_path = os.path.join(local_dir, 'dataset.yaml')
dataset_yaml_content = {
    'path': local_dir,
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
with open(dataset_yaml_path, 'w') as file:
    yaml.dump(dataset_yaml_content, file, default_flow_style=False)

# Here is where manual hyperparameters are defined for the YOLOv5 model
# These were carefully selected to optimize the model for the dataset
hyp = {
    'optimizer': 'adam',
    'mosaic': 1.0,
    'mixup': 0.2,
    'jitter': 0.2,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 0.2,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 0.1
}
hyp_yaml_path = 'data/hyps/hyp.custom.yaml'
with open(hyp_yaml_path, 'w') as file:
    yaml.dump(hyp, file)

# The directory for the training phase is in the 'Runs/YOLO_Training' directory
project_dir = os.path.join(local_dir, 'Runs/YOLO_Training')
# Here we use the run_command() function to execute the shell command to train the YOLOv5 model
# THe name of the model is 'Final_YOLOv5_Model' and is where the weights and other images, plots, etc. will be saved
run_command(f"python train.py --img 640 --batch 16 --epochs 300 \
--data '{dataset_yaml_path}' \
--hyp '{hyp_yaml_path}' \
--weights yolov5s.pt \
--cache \
--project '{project_dir}' \
--name Final_YOLOv5_Model")

# here we define the project directory for the testing phase
test_project_dir = os.path.join(local_dir, 'Runs/Test')
# We can use the trained model to test on the test dataset
run_command(f"python val.py \
--weights '{project_dir}/Final_YOLOv5_Model/weights/best.pt' \
--data '{dataset_yaml_path}' \
--img 640 \
--batch 32 \
--task test \
--project '{test_project_dir}' \
--name Final_YOLOv5_Test \
--exist-ok")
