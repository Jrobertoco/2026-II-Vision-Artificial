import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers
import numpy as np


###########################
import cv2

def agregar_clahe(images):
    nuevas = []

    for img in images:

        gray = cv2.cvtColor(
            (img*255).astype(np.uint8),
            cv2.COLOR_RGB2GRAY
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8,8)
        )

        canal = clahe.apply(gray)
        canal = canal.astype(np.float32)/255.0

        canal = np.expand_dims(canal, axis=-1)

        nuevas.append(
            np.concatenate([img, canal], axis=-1)
        )

    return np.array(nuevas)
###########################

keras.utils.set_random_seed(42)

# ==========================
# CARGA DE DATOS
# ==========================


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(
    label_mode="fine"
)

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train_clahe = agregar_clahe(x_train)
x_test_clahe = agregar_clahe(x_test)

print(x_train_clahe.shape)
print(y_train.shape)

# ==========================
# MODELO
# ==========================



#
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(
        height_factor=0.1,
        width_factor=0.1
    )
])
#
model = keras.Sequential([

    keras.Input(shape=(32,32,4)),
    data_augmentation,
    

    layers.Conv2D(
        64,#32,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),
    layers.BatchNormalization(),

    layers.Conv2D(
        64,#32,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),
    layers.BatchNormalization(),

    layers.MaxPooling2D(),

    layers.Conv2D(
        128,#64,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),
    layers.BatchNormalization(),

    layers.Conv2D(
        128,#64,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),
    layers.BatchNormalization(),

    layers.MaxPooling2D(),

    layers.Conv2D(
        192,#128,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),

        layers.Conv2D(
        192,#128,
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
    x_train_clahe,
    y_train,
    batch_size=128,
    epochs=30,
    validation_split=0.1
)

# ==========================
# EVALUACION
# ==========================

test_loss, test_acc = model.evaluate(
    x_test_clahe,
    y_test,
    verbose=0
)

print(f"Test Accuracy: {test_acc:.4f}")
