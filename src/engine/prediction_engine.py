import json
import os

from tensorflow.keras.models import load_model


def load_trained_model():
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../keras-models'))
    model_path = os.path.join(model_dir, 'efficient_net_model_RC2')

    print('path loaded', model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model folder not found: {model_path}")

    # load all model
    loaded_model = load_model(model_path)

    return loaded_model


def load_classes():
    script_dir = os.path.dirname(__file__)
    classes_path = os.path.join(script_dir, '../keras-models/classes.json')

    with open(classes_path, 'r') as f:
        class_mapping = json.load(f)
    return class_mapping


def make_prediction(model, data):
    prediction = model.predict(data)
    return prediction.tolist()
