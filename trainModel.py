import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

department_num = 10
max_tokens = 8000

df = pd.read_csv("datos.csv", quoting=1)

X = df["Entrada"].values
y_department = pd.get_dummies(df["Departamento"])  # One-hot para 10 departamentos
y_priority = df["Prioridad"].values.reshape(-1, 1)

# 3. División train/val (90%/10%)
X_train, X_val, y_dept_train, y_dept_val, y_pri_train, y_pri_val = train_test_split(
    X, 
    y_department, 
    y_priority, 
    test_size=0.1,  # 10% para validación
    stratify=y_department,  # Mantener distribución balanceada
    random_state=42
)

# 4. Crear datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    {"text_input": X_train},
    {
        "department_output": y_dept_train.astype("float32"),  # Keras requiere float32
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

# Configuración del vectorizador
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens= max_tokens,  # Reducido por el tamaño del dataset (1800 muestras)
    output_sequence_length=64,  # 64 palabras máximo
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    output_mode="int"
)
vectorizer.adapt(X_train) 

# Entrada
text_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="text_input")

# Capas de procesamiento
x = vectorizer(text_input)
x = tf.keras.layers.Embedding(input_dim= max_tokens + 1, output_dim=96, name="embedding")(x)  # +1 para OOV
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)  # Regularización para evitar overfitting

# Salidas
department_output = tf.keras.layers.Dense(
    10,  # 10 departamentos
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
        "department_output": 0.5,  # Ambas tareas son igual de importantes
        "priority_output": 0.5
    },
    metrics={
        "department_output": ["accuracy"],
        "priority_output": [
            tf.keras.metrics.MeanAbsoluteError(),  # Error absoluto promedio
            tf.keras.metrics.MeanSquaredError()  # Para monitorear outliers
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
    batch_size=32,  # Tamaño manejable en CPU
    callbacks=[early_stopping]
)