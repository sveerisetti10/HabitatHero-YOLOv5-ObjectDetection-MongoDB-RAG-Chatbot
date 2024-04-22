![Marvel](flask_app/app/static/images/wildlife.jpeg "Wildlife")

# HabitatHero

## Purpose

This project aims to raise public awareness and foster conservation engagement regarding endangered species, which are often at the risk of extinction due to inadequate understanding and engagement in conservation efforts. Despite the presence of conservation tools, there's a gap in educational resources that provide a comprehensive understanding of the threats these species face and the actions we can take to support their survival. The International Union for Conservation of Nature (IUCN) reports that at least 38,500 species are threatened, with over 16,300 endangered. Our initiative addresses this issue by educating the public about various wildlife animals and ways humans can help ensure their prosperity.

The model focuses on the following wildlife animals: antelope, bat, bear, bee, butterfly, Domestic short-haired cats, chimpanzee, coyote, dolphin, eagle, elephant, gorilla, hippopotamus, rhinoceros, hummingbird, kangaroo, koala, leopard, lion, lizard, orangutan, panda, penguin, seal, shark, tiger, turtle, whale, and zebra.

The app leverages computer vision to identify and detect these animals, drawing inspiration from influential research in the field, such as “The iNaturalist Species Classification and Detection Dataset” by Grant Van Horn et al.

You can visit the application here: https://flask-habitathero-app-al7e4bbgba-uc.a.run.app

## Flask App Structure

The Flask application comprises a front-end designed in HTML/CSS & JavaScript, and a Python back-end. Key components include:

- `app/static/`: Contains CSS and images for the application's UI.
- `app/templates/`: Hosts HTML for the main user interface and results display.
- `app/__init__.py`: Initializes the Flask app instance and configures the OpenAI key.
- `app/inference.py`: Executes the trained YOLOv5 model for wildlife image detection.
- `app/rag_model.py`: Integrates OpenAI's API, MongoDB, and Flask for intelligent backend processing and interaction.
- `app/routes.py`: Defines the Flask app routes.
- `Dockerfile`: Constructs the Docker image for deployment on Google Cloud Run.
- `Procfile`: Specifies the gunicorn server for the Flask application.
- `requirements.txt`: Lists all dependencies for the application.
- `run.py`: Starts the Flask web application for production.

## Models

- `SVM_classical.py`: Trains an SVM model using HOG features to classify animal images.
- `naive_YOLOv5.py`: Trains a YOLOv5 model with the first 24 layers frozen.
- `main_YOLOv5.py`: Trains the YOLOv5 model and performs inference with optimal weights.

## Notebooks

This directory contains various Jupyter Notebooks for exploratory code, model training, and RAG implementation.

## Setup and Requirements

- `requirements.txt`: A requirements file to document project dependencies.
- `setup.py`: A setup script that prepares the project environment, such as retrieving data, configuring YOLOv5 for training, and executing the training and inference process.

## Getting Started

To begin using this project, clone the repository and ensure you have the required dependencies installed by running:

```bash
pip install -r requirements.txt
