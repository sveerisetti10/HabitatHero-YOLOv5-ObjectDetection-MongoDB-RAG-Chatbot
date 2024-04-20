
import os
import torch
from PIL import Image as PILImage
from io import BytesIO

# Here we define the classes that the object detection model can detect (based on the dataset used for training)
class_names = [
    'antelope', 'bat', 'bear', 'butterfly', 'Domestic short-haired cats',
    'chimpanzee', 'coyote', 'dolphin', 'eagle',
    'elephant', 'gorilla', 'hippopotamus', 'rhinoceros',
    'hummingbird', 'kangaroo', 'koala', 'leopard', 'lion',
    'lizard', 'orangutan', 'panda', 'penguin', 'seal',
    'shark', 'tiger', 'turtle', 'whale', 'zebra', 'bee'
]

# Here we use the model weights to load the model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='app/model_weights/best.pt')

def run_inference(image_bytes):
    """
    Run model inference on the uploaded image and return the image with bounding boxes
    and the names of detected animals.
    """
    # Convert bytes to a PIL Image
    image = PILImage.open(BytesIO(image_bytes))

    # Here we perform inference on the image using the YOLOv5 model
    results = model(image)
    
    # Here we create an image with bounding boxes
    img_with_boxes = results.render()[0]  
    
    # Within the results, we want to access the detected classes
    detected_classes = []
    # The for loop iterates over the detected objects and gets the class name
    for *xyxy, conf, cls in results.xyxy[0]:
        detected_class_name = class_names[int(cls.item())]
        # Here we append the class name to the detected_classes list
        detected_classes.append(detected_class_name)

    # Here we convert the image with bounding boxes to bytes so that it can be displayed on the HTML page
    img_bytes = BytesIO()
    # Here we save the image with bounding boxes to the img_bytes object in the JPEG format
    PILImage.fromarray(img_with_boxes).save(img_bytes, format='JPEG')
    # Here we return the image bytes and the detected classes
    return img_bytes.getvalue(), detected_classes
