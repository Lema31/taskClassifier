from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
import os
import json

with open("department_order.json", "r", encoding="utf-8") as f:
    department_names = json.load(f)

model = tf.keras.models.load_model("taskClassifierModel.keras")

app = Flask(__name__)
CORS(app)

@app.get('/healthCheck')
def healthCheck():
    return 'Ok'

@app.post('/taskClassifier')
def getPhylum():
    data = request.get_json()
    task = data['task']

    input_data = tf.constant([task])
    department_probs, priority = model.predict(input_data)
    department_index = np.argmax(department_probs)
    department_predicted = department_names[department_index]
    # Respuesta
    return jsonify({
        "department": department_predicted,
        "priority": str(round(priority[0][0] * 100))
    })

if __name__ == "__main__":
    load_dotenv()
    port = int(os.getenv("PORT"))
    app.run(debug = True, port = port, host= "0.0.0.0")