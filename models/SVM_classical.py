

import os
import numpy as np
from skimage import io, color, transform
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import concurrent.futures
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_image(img_name, image_path, annotation_path, classes):
    """
    Purpose: Preprocess an image and its annotation.
    Input: img_name - name of the image file
    Input: image_path - path to the directory containing the images
    Input: annotation_path - path to the directory containing the annotations
    Input: classes - list of class names
    """
    base_name = os.path.splitext(img_name)[0]
    annotation_file = os.path.join(annotation_path, base_name + '.txt')
    if not os.path.exists(annotation_file):
        return None
    
    # Here we open the annotation file of the image 
    with open(annotation_file, 'r') as file:
        # Here we arre trying to get the class id of the image which is the first element in the first line of the annotation file
        class_id = int(file.readline().split()[0])
        # Here we are checking if the class id is greater than the length of the classes list
        if class_id >= len(classes):
            return None
        
    # Here we define the image path
    img_path = os.path.join(image_path, img_name)
    img = io.imread(img_path)
    # Here we are checking if the image has 4 channels and if it does we convert it to 3 channels
    if img.ndim == 3 and img.shape[2] == 4:
        # Here we are converting the image to 3 channels using the rgba2rgb function
        img = color.rgba2rgb(img)
    img_resized = transform.resize(img, (128, 64))

    # Here we are checking if the image has 2 dimensions and if it does we get the hog features
    if img_resized.ndim == 2:
        # Here we use the hog function to get the hog features of the image
        hog_features = hog(img_resized, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False)
    else:
        # Here we use the hog function to get the hog features of the image
        hog_features = hog(img_resized, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False, channel_axis=-1)
    # Here we return the hog features and the class id for the picture 
    return hog_features, class_id

def load_image_label_pairs(image_path, annotation_path, classes):
    """
    Purpose: Load image and label pairs from the image and annotation directories.
    Input: image_path - path to the directory containing the images
    Input: annotation_path - path to the directory containing the annotations
    Input: classes - list of class names
    """
    data, labels = [], []
    img_names = os.listdir(image_path)
    # Here we use the ThreadPoolExecutor to preprocess the images concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Here we use the submit function to submit the preprocess_image function to the executor
        futures = {executor.submit(preprocess_image, img, image_path, annotation_path, classes): img for img in img_names}
        # Here we use the as_completed function to iterate over the futures as they are completed and we can monitor the progress using the tqdm function
        progress = tqdm(concurrent.futures.as_completed(futures), total=len(img_names), desc="Processing Images", unit="img")
        # Here we iterate over the futures as they are completed
        for future in progress:
            # The result method is used to get the result of the future
            result = future.result()
            # Here we are checking if the result is not None
            if result:
                # If there is a result, we gather the hog features and the class id
                hog_features, class_id = result
                # We can append the hog features to the empty data list and the class id to the empty labels list
                data.append(hog_features)
                labels.append(class_id)
        progress.close()
    # Here we return the data and labels as numpy arrays
    return np.array(data), np.array(labels)

# Here we define the classes 
classes = [
    'antelope', 'bat', 'bear', 'butterfly', 'Domestic short-haired cats', 'chimpanzee', 'coyote', 'dolphin', 'eagle',
    'elephant', 'gorilla', 'hippopotamus', 'rhinoceros', 'hummingbird', 'kangaroo', 'koala', 'leopard', 'lion',
    'lizard', 'orangutan', 'panda', 'penguin', 'seal', 'shark', 'tiger', 'turtle', 'whale', 'zebra', 'bee'
]

# Here we need to define the path to the image folder as well as the annotation folder 
image_path = '/path/to/image_folder'
annotation_path = '/path/to/annotation_folder'

# Here the X is essentially a vector of hog features and y is a vector of class ids
X, y = load_image_label_pairs(image_path, annotation_path, classes)
# We can then use the train_test_split function to split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# We can use SVM to train the model
svm = SVC(kernel='linear', C=1.0, random_state=42)
# Fit the model to the training data 
svm.fit(X_train, y_train)
# Make the prediction on the test data
y_pred = svm.predict(X_test)
# Print out the accuracy and classification report for more information 
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=classes))

# Here we create a confusion matrix to create the heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
# Output the heatmap
plt.show()
