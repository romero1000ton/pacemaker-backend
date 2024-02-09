from flask import Flask
from flask_cors import CORS

from src.routes.prediction_routes import prediction_routes

app = Flask(__name__)
CORS(app)
app.register_blueprint(prediction_routes)

if __name__ == '__main__':
    app.run()