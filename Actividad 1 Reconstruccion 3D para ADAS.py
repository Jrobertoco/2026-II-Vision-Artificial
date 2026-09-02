#pip install opencv-python numpy matplotlib
#pip install opencv-python numpy matplotlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog


# Seleccion de imagen para trabajar


root = tk.Tk()
root.withdraw()

ruta_imagen = filedialog.askopenfilename(
    title="Selecciona una imagen",
    filetypes=[
        ("Imágenes", "*.jpg *.jpeg *.png *.bmp")
    ]
)

if not ruta_imagen:
    print("No se seleccionó ninguna imagen.")
    exit()


# Cargar imagen y guardarla en variable img

img = cv2.imread(ruta_imagen)

if img is None:
    print("No fue posible abrir la imagen.")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# MÉTODO 1: CIELAB 

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# Separar canales Iluminancia (L), a* y b*
L, A, B = cv2.split(lab)

# Máscara para ROJO usando a*
mask_lab_red = (A > 150).astype(np.uint8) * 255

# Máscara para AMARILLO usando b*
mask_lab_yellow = (B > 145).astype(np.uint8) * 255

# Recolorear región roja detectada
result_lab_red = img_rgb.copy()
result_lab_red[mask_lab_red > 0] = [0, 255, 0]

# Recolorear región amarilla detectada
result_lab_yellow = img_rgb.copy()
result_lab_yellow[mask_lab_yellow > 0] = [0, 0, 255]


# MÉTODO 2: HSV

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Detección clásica del rojo mediante matiz H
lower_red1 = np.array([0, 80, 50])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 80, 50])
upper_red2 = np.array([180, 255, 255])

mask_hsv_red_1 = cv2.inRange(
    hsv,
    lower_red1,
    upper_red1
)

mask_hsv_red_2 = cv2.inRange(
    hsv,
    lower_red2,
    upper_red2
)

mask_hsv_red = cv2.bitwise_or(
    mask_hsv_red_1,
    mask_hsv_red_2
)

# Detección clásica del amarillo mediante matiz H
lower_yellow = np.array([20, 80, 80])
upper_yellow = np.array([35, 255, 255])

mask_hsv_yellow = cv2.inRange(
    hsv,
    lower_yellow,
    upper_yellow
)

# Recolorear detección HSV rojo
result_hsv_red = img_rgb.copy()
result_hsv_red[mask_hsv_red > 0] = [0, 255, 0]

# Recolorear detección HSV amarillo
result_hsv_yellow = img_rgb.copy()
result_hsv_yellow[mask_hsv_yellow > 0] = [0, 0, 255]


# Métricas simples para comparación

pixeles_lab_red = np.sum(mask_lab_red > 0)
pixeles_hsv_red = np.sum(mask_hsv_red > 0)

pixeles_lab_yellow = np.sum(mask_lab_yellow > 0)
pixeles_hsv_yellow = np.sum(mask_hsv_yellow > 0)

print("\nRESULTADOS")

print("\nROJO")
print("Pixeles LAB:", pixeles_lab_red)
print("Pixeles HSV:", pixeles_hsv_red)

print("\nAMARILLO")
print("Pixeles LAB:", pixeles_lab_yellow)
print("Pixeles HSV:", pixeles_hsv_yellow)


# Mostrar resultados

plt.figure(figsize=(18, 12))

plt.subplot(3, 3, 1)
plt.imshow(img_rgb)
plt.title("Imagen Original")
plt.axis("off")

plt.subplot(3, 3, 2)
plt.imshow(mask_lab_red, cmap="gray")
plt.title("Mascara CIELAB Rojo")
plt.axis("off")

plt.subplot(3, 3, 3)
plt.imshow(result_lab_red)
plt.title("CIELAB Rojo -> Verde")
plt.axis("off")

plt.subplot(3, 3, 5)
plt.imshow(mask_hsv_red, cmap="gray")
plt.title("Mascara HSV Rojo")
plt.axis("off")

plt.subplot(3, 3, 6)
plt.imshow(result_hsv_red)
plt.title("HSV Rojo -> Verde")
plt.axis("off")

plt.subplot(3, 3, 8)
plt.imshow(mask_lab_yellow, cmap="gray")
plt.title("Mascara CIELAB Amarillo")
plt.axis("off")

plt.subplot(3, 3, 9)
plt.imshow(result_lab_yellow)
plt.title("CIELAB Amarillo -> Azul")
plt.axis("off")

plt.tight_layout()
plt.show()