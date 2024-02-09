import numpy as np

from src.engine.prediction_engine import make_prediction, load_classes


def predict(data, loaded_model):
    prediction = make_prediction(loaded_model, data)
    class_mapping = load_classes()
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_mapping[str(predicted_class_index)]
    confidence = prediction[0][predicted_class_index]
    response = {
        'predicted_class': predicted_class,
        'confidence': round(confidence * 100, 2)
    }
    return response
