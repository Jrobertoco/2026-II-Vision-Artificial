import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers
import numpy as np

keras.utils.set_random_seed(42)

# ==========================
# CARGA DE DATOS
# ==========================

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(
    label_mode="fine"
)

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print(x_train.shape)
print(y_train.shape)

# ==========================
# MODELO
# ==========================

model = keras.Sequential([

    keras.Input(shape=(32,32,3)),

    layers.Conv2D(
        32,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),
    layers.BatchNormalization(),

    layers.Conv2D(
        32,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),
    layers.BatchNormalization(),

    layers.MaxPooling2D(),

    layers.Conv2D(
        64,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),
    layers.BatchNormalization(),

    layers.Conv2D(
        64,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),
    layers.BatchNormalization(),

    layers.MaxPooling2D(),

    layers.Conv2D(
        128,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),
    layers.BatchNormalization(),

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.3),

    layers.Dense(
        100,
        activation="softmax"
    )
])

model.summary()

# ==========================
# COMPILACION
# ==========================

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================
# ENTRENAMIENTO
# ==========================

history = model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=30,
    validation_split=0.1
)

# ==========================
# EVALUACION
# ==========================

test_loss, test_acc = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print(f"Test Accuracy: {test_acc:.4f}")
