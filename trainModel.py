import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import json

department_num = 10
max_tokens = 8000

df = pd.read_csv("datos.csv", quoting=1)

X = df["Entrada"].values
y_department = pd.get_dummies(df["Departamento"])  
y_priority = df["Prioridad"].values.reshape(-1, 1)

class_names = y_department.columns.tolist()


with open("department_order.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f)


X_train, X_val, y_dept_train, y_dept_val, y_pri_train, y_pri_val = train_test_split(
    X, 
    y_department, 
    y_priority, 
    test_size=0.1,  
    stratify=y_department,  
    random_state=42
)

train_dataset = tf.data.Dataset.from_tensor_slices((
    {"text_input": X_train},
    {
        "department_output": y_dept_train.astype("float32"),
        "priority_output": y_pri_train.astype("float32")
    }
)).batch(32).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((
    {"text_input": X_val},
    {
        "department_output": y_dept_val.astype("float32"),
        "priority_output": y_pri_val.astype("float32")
    }
)).batch(32)


vectorizer = tf.keras.layers.TextVectorization(
    max_tokens= max_tokens,  
    output_sequence_length=64,  
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    output_mode="int"
)
vectorizer.adapt(X_train) 


text_input = tf.keras.Input(shape=(), dtype=tf.string, name="text_input")


x = vectorizer(text_input)
x = tf.keras.layers.Embedding(input_dim= max_tokens + 1, output_dim=96, name="embedding")(x)  
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)  


department_output = tf.keras.layers.Dense(
    10,  
    activation="softmax",
    name="department_output"
)(x)

priority_output = tf.keras.layers.Dense(
    1,
    activation="sigmoid",
    name="priority_output"
)(x)

model = tf.keras.Model(text_input, [department_output, priority_output])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={
        "department_output": tf.keras.losses.CategoricalCrossentropy(),
        "priority_output": tf.keras.losses.MeanSquaredError()
    },
    loss_weights={
        "department_output": 0.5,  
        "priority_output": 0.5
    },
    metrics={
        "department_output": ["accuracy"],
        "priority_output": [
            tf.keras.metrics.MeanAbsoluteError(),  
            tf.keras.metrics.MeanSquaredError()  
        ]
    }
)

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# Entrenar
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    batch_size=32,  
    callbacks=[early_stopping]
)

model.save("taskClassifierModel.keras")