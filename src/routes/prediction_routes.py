import io

import numpy as np
from PIL import Image
from flask import Blueprint, request, jsonify
from tensorflow.keras.preprocessing import image

from src.controllers.prediction_controller import predict
from src.engine.prediction_engine import load_trained_model

prediction_routes = Blueprint('prediction_routes', __name__)
loaded_model = load_trained_model()

IMG_SIZE = 224


@prediction_routes.route('/predict', methods=['POST'])
def predict_route():
    image_request = request.files['imageData']
    image_data = image_request.read()
    image_read = Image.open(io.BytesIO(image_data))
    img = image_read.resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = predict(img_array, loaded_model)

    return jsonify(prediction)


@prediction_routes.route('/home', methods=['GET'])
def home_route():
    return 'Soy la home'
