import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers
import numpy as np


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

keras.utils.set_random_seed(42)


fine_labels = [
'apple','aquarium_fish','baby','bear','beaver',
'bed','bee','beetle','bicycle','bottle',
'bowl','boy','bridge','bus','butterfly',
'camel','can','castle','caterpillar','cattle',
'chair','chimpanzee','clock','cloud','cockroach',
'couch','crab','crocodile','cup','dinosaur',
'dolphin','elephant','flatfish','forest','fox',
'girl','hamster','house','kangaroo','computer_keyboard',
'lamp','lawn_mower','leopard','lion','lizard',
'lobster','man','maple_tree','motorcycle','mountain',
'mouse','mushroom','oak_tree','orange','orchid',
'otter','palm_tree','pear','pickup_truck','pine_tree',
'plain','plate','poppy','porcupine','possum',
'rabbit','raccoon','ray','road','rocket',
'rose','sea','seal','shark','shrew',
'skunk','skyscraper','snail','snake','spider',
'squirrel','streetcar','sunflower','sweet_pepper','table',
'tank','telephone','television','tiger','tractor',
'train','trout','tulip','turtle','wardrobe',
'whale','willow_tree','wolf','woman','worm'
]
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
y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred, axis=1)

cm = confusion_matrix(
    y_test.flatten(),
    y_pred
)

# Visualize the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show




cm_norm = cm.astype(float) / cm.sum(axis=1)[:,None]
plt.savefig(
    "confusion_matrix_1.png",
    dpi=300,
    bbox_inches="tight"
)
plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=False,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(
    "confusion_matrix_2.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

errores = np.where(
    y_pred != y_test.flatten()
)[0]


class_acc = []

for i in range(100):
    correctos = cm[i, i]
    total = np.sum(cm[i, :])
    acc = correctos / total
    class_acc.append(acc)
class_acc = np.array(class_acc)

best_class = np.argmax(class_acc)
worst_class = np.argmin(class_acc)

print("Mejor clase:", fine_labels[best_class])
print("Accuracy:", class_acc[best_class])

print("Peor clase:", fine_labels[worst_class])
print("Accuracy:", class_acc[worst_class])


best10 = np.argsort(class_acc)[-10:][::-1]
worst10 = np.argsort(class_acc)[:10]

print("\nTOP 10 MEJORES")

for c in best10:
    print(
        f"Clase {c}: {fine_labels[c]} - {class_acc[c]*100:.2f}%"
    )

print("\nTOP 10 PEORES")

for c in worst10:
    print(
        f"Clase {c}: {fine_labels[c]} - {class_acc[c]*100:.2f}%"
    )


    print(
    f"Mejor clase: {fine_labels[best_class]}"
)

print(
    f"Peor clase: {fine_labels[worst_class]}"
)