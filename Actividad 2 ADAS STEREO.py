import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Seleccion de imagen izquierda y derecha para trabajar


root = tk.Tk()
root.withdraw()

ruta_imagen = filedialog.askopenfilename(
    title="Selecciona la imagen izquierda",
    filetypes=[
        ("Imágenes", "*.jpg *.jpeg *.png *.bmp")
    ]
)

if not ruta_imagen:
    print("No se seleccionó ninguna imagen.")
    exit()


# Cargar imagen y guardarla en variable img

left = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

root = tk.Tk()
root.withdraw()

ruta_imagen = filedialog.askopenfilename(
    title="Selecciona la imagen derecha",
    filetypes=[
        ("Imágenes", "*.jpg *.jpeg *.png *.bmp")
    ]
)

if not ruta_imagen:
    print("No se seleccionó ninguna imagen.")
    exit()


# Cargar imagen y guardarla en variable img_right

#cv2.imread("imagen.png", 0) # Escala de grises

#cv2.imread("imagen.png", 1) # Color (por defecto)

#cv2.imread("imagen.png", -1) # Mantiene canales y transparencia (alpha)


right = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)


print("Imagen izquierda:", left.shape)
print("Imagen derecha:", right.shape)
if left.shape != right.shape:
    print("Las imágenes deben tener el mismo tamaño")
exit()


#####
def compute_sad_disparity(left, right, window_size=7, max_disparity=64):

    # Convertir una sola vez
    left = left.astype(np.int32)
    right = right.astype(np.int32)

    h, w = left.shape
    offset = window_size // 2

    disparity = np.zeros((h, w), dtype=np.uint8)

    for y in range(offset, h - offset):

        # Mostrar avance cada 50 filas
        if y % 50 == 0:
            print(f"Procesando fila {y}/{h}")

        for x in range(offset + max_disparity, w - offset):

            best_cost = float("inf")
            best_d = 0

            window_left = left[
                y - offset:y + offset + 1,
                x - offset:x + offset + 1
            ]

            for d in range(max_disparity):

                xr = x - d

                if xr - offset < 0:
                    continue

                window_right = right[
                    y - offset:y + offset + 1,
                    xr - offset:xr + offset + 1
                ]

                sad = np.sum(
                    np.abs(window_left - window_right)
                )

                if sad < best_cost:
                    best_cost = sad
                    best_d = d

            disparity[y, x] = int(
                best_d * 255 / max_disparity
            )

    return disparity



left = cv2.resize(left, (640, 360))
right = cv2.resize(right, (640, 360))

disp_sad = compute_sad_disparity(
    left,
    right,
    #window_size=7

    window_size = 5,
    max_disparity = 32,
    )


def mostrar_imagenes(left, right, disparity, title):
    plt.figure(figsize=(12,5))

    plt.subplot(1,3,1)
    plt.imshow(left, cmap='gray')
    plt.title("Left")

    plt.subplot(1,3,2)
    plt.imshow(right, cmap='gray')
    plt.title("Right")

    plt.subplot(1,3,3)
    plt.imshow(disparity, cmap='jet')
    plt.title(title)

    plt.show()


mostrar_imagenes(left, right, disp_sad, "Disparidad SAD")

def compute_ssd_disparity(left, right, window_size=5, max_disparity=32):

    left = left.astype(np.int32)
    right = right.astype(np.int32)

    h, w = left.shape
    offset = window_size // 2

    disparity = np.zeros((h, w), dtype=np.uint8)

    for y in range(offset, h - offset):

        if y % 50 == 0:
            print(f"Procesando fila {y}/{h}")

        for x in range(offset + max_disparity, w - offset):

            best_cost = float("inf")
            best_d = 0

            window_left = left[
                y-offset:y+offset+1,
                x-offset:x+offset+1
            ]

            for d in range(max_disparity):

                xr = x - d

                if xr - offset < 0:
                    continue

                window_right = right[
                    y-offset:y+offset+1,
                    xr-offset:xr+offset+1
                ]

                ssd = np.sum(
                    (window_left - window_right) ** 2
                )

                if ssd < best_cost:
                    best_cost = ssd
                    best_d = d

            disparity[y, x] = best_d

    return disparity



mostrar_imagenes(left, right, disp_sad, "Disparidad SAD")

disp_ssd = compute_ssd_disparity(
    left,
    right,
    window_size = 5,
    max_disparity = 32,
    )

mostrar_imagenes(left, right, disp_ssd, "Disparidad SSD")


#cambiar gamma de la imagen derecha para mejorar el calculo de dispariedad
gamma = 0.5
lut = np.array(
[
((i / 255.0) ** gamma) * 255
for i in range(256)
]).astype("uint8")
right_gamma = cv2.LUT(right, lut)


disp_gamma_sad = compute_sad_disparity(left, right_gamma, window_size=5, max_disparity=32)
mostrar_imagenes(left, right_gamma, disp_gamma_sad, "Disparidad SAD con gamma")

disp_gamma_ssd = compute_ssd_disparity(left, right_gamma, window_size=5, max_disparity=32)
mostrar_imagenes(left, right_gamma, disp_gamma_ssd, "Disparidad SSD con gamma")

#Transformada Census
def census_transform(img, window_size=5):

    h, w = img.shape
    offset = window_size // 2

    census = np.zeros((h, w), dtype=np.uint64)

    for y in range(offset, h - offset):

        if y % 50 == 0:
            print(f"Census fila {y}/{h}")

        for x in range(offset, w - offset):

            center = img[y, x]

            descriptor = 0

            for j in range(-offset, offset + 1):
                for i in range(-offset, offset + 1):

                    if i == 0 and j == 0:
                        continue

                    descriptor <<= 1

                    if img[y + j, x + i] >= center:
                        descriptor |= 1

            census[y, x] = descriptor

    return census


#Distancia de Hamming
def hamming_distance(a, b):
    return (a ^ b).bit_count()

#calculo de disparidad usando Census y Hamming
def compute_census_disparity(
        left,
        right,
        window_size=5,
        max_disparity=32):

    h, w = left.shape
    offset = window_size // 2

    print("Calculando Census Left...")
    census_left = census_transform(left, window_size)

    print("Calculando Census Right...")
    census_right = census_transform(right, window_size)

    disparity = np.zeros((h, w), dtype=np.uint8)

    for y in range(offset, h - offset):

        if y % 50 == 0:
            print(f"Matching fila {y}/{h}")

        for x in range(offset + max_disparity, w - offset):

            best_cost = float("inf")
            best_d = 0

            for d in range(max_disparity):

                xr = x - d

                if xr < offset:
                    continue

                cost = hamming_distance(
                    int(census_left[y, x]),
                    int(census_right[y, xr])
                )

                if cost < best_cost:
                    best_cost = cost
                    best_d = d

            disparity[y, x] = best_d

    return disparity


disp_census = compute_census_disparity(left, right, window_size=5, max_disparity=32)
mostrar_imagenes(left, right, disp_census, "Disparidad Census")

disp_census = compute_census_disparity(left, right_gamma, window_size=5, max_disparity=32)
mostrar_imagenes(left, right_gamma, disp_census, "Disparidad Census")