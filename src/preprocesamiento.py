import cv2
import os

def preprocesar_imagen(ruta):
 """
 Lee una imagen y aplica las primeras etapas del pipeline:
 - Escala de grises
 - Blur
 - Detección de bordes
 - Redimensionado
 """

 #Se comprueba si la ruta existe
 if not os.path.exists(ruta):
    print(f"⚠️ Archivo no encontrado: {ruta}") 
    return None
 
 #Lee la imagen desde disco y devuelve un array de NumPy 
 #con los valores de los pixeles
 img = cv2.imread(ruta) 

 #Comprueba que el archivo sea válido
 if img is None:
    print(f"⚠️ Imagen corrupta o no válida: {ruta}") 
    return None
 
 #Convierte la imagen a escala de grises, reduciendo información redundante de color.
 #Esto simplifica el análisis y reduce el tamaño de los datos.
 gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

 #Redimensiona la imagen a 64x64 píxeles
 resized = cv2.resize(gris, (64, 64))

 return resized #Devuelve la imagen preprocesada