from flask import current_app, Blueprint, render_template, request, url_for
from werkzeug.utils import secure_filename
import os
from .inference import run_inference
from flask import request, jsonify
from .rag_model import get_animal_info
from flask import session
from .rag_model import get_response_for_query

main = Blueprint('main', __name__)

# Here we define the main route to the application 
@main.route('/')
def home():
    """
    Purpose: Render the main page of the application
    """
    return render_template('index.html')

# Here we define another route to upload an image
@main.route('/upload', methods=['POST'])
def upload_image():
    """
    Purpose: Upload an image, run inference, and display the results
    """
    # Check if an image was uploaded
    if 'image' in request.files:
        image = request.files['image']
        if image.filename != '':
            # Here we secure the filename to prevent any malicious attacks
            filename = secure_filename(image.filename)
            # Here we save the image to the static directory
            filepath = os.path.join(current_app.root_path, 'static/images', filename)
            image.save(filepath)

            # We open the file and then read the image bytes
            with open(filepath, 'rb') as file:
                image_bytes = file.read()
                # We use the run_inference function to get the output image and detected classes
                # This function is defined in the 
                output_image_bytes, detected_classes = run_inference(image_bytes)
            
            # Here we save the processed image to the static directory
            output_filename = f"processed_{filename}"
            output_path = os.path.join(current_app.root_path, 'static/images', output_filename)
            with open(output_path, 'wb') as f:
                f.write(output_image_bytes)
            
            # Get URL for the processed image
            image_url = url_for('static', filename=f'images/{output_filename}')
            
            # Render results template
            return render_template('results.html', image_url=image_url, detected_classes=detected_classes)
    # If the image is not a wildlife animal that falls within the categories, we return an error. 
    return 'No image uploaded', 400

# Here we create a route to get information about the detected animal
@main.route('/get-animal-info', methods=['POST'])
def animal_info():
    """
    Purpose: Get information about the detected animal
    """
    data = request.json
    # Here we get the detected class from the request data
    detected_class = data.get('animal_name')
    # If there is a detected class, we use the get_animal_info function to get information about the animal
    # The get_animal_info function is defined in the rag_model.py file
    if detected_class:
        info = get_animal_info(detected_class)
        return jsonify(info=info)
    else:
        return jsonify({'error': 'No animal class provided'}), 400

# Here we create a route to chat with the chatbot 
@main.route('/chat', methods=['POST'])
def chat():
    """
    Chat with the chatbot using a more context-aware response generation
    """
    data = request.json
    message = data.get('message', '')

    if 'conversation' not in session:
        session['conversation'] = []

    # Append the user's message to the conversation history
    session['conversation'].append({'role': 'user', 'content': message})

    # Generate the chat response using the new function which uses embeddings and contextual understanding
    response_message = get_response_for_query(message)

    # Append the assistant's response to the conversation history
    session['conversation'].append({'role': 'assistant', 'content': response_message})

    # Return the response message as JSON
    return jsonify({'message': response_message})
