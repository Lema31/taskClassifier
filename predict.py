import numpy as np
import tensorflow as tf
import json


model = tf.keras.models.load_model("taskClassifierModel.keras")



with open("department_order.json", "r", encoding="utf-8") as f:
    department_names = json.load(f)



input_data = tf.constant(['Arreglar problema urgente en la API de la empresa'])

department_probs, priority = model.predict(input_data)
department_index = np.argmax(department_probs)
print(department_names[department_index])
print(priority)