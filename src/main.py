import os
import numpy as np

from preprocesamiento import preprocesar_imagen
from caracteristicas import extraer_hog
from modelo import entrenar_modelo
from evaluacion import evaluar_modelo

X = [] #Lista que guardará los vectores de caract. HOG
y = [] #Lista que guardará las etiquetas de clase

#Este fragmento de código se encarga de recorrer todas las imagenes
#de tornillos y tuercas, convertir cada imagen en números y guardar 
#esos números en X y su clase en Y
for etiqueta, clase in enumerate(["gatos", "perros"]):
 carpeta = f"../datos/{clase}"
 for archivo in os.listdir(carpeta):
    ruta = os.path.join(carpeta, archivo)

    img_proc = preprocesar_imagen(ruta)
    if img_proc is not None:
        feat = extraer_hog(img_proc)
        X.append(feat)
        y.append(etiqueta)

#Convertir las listas en arrays de NumPy
X = np.array(X)
y = np.array(y)

# Entrenar modelo
modelo, X_test, y_test = entrenar_modelo(X, y)

# Evaluar modelo
evaluar_modelo(modelo, X_test, y_test)

# Probar o predecir una imagen nueva

# Carpeta donde estarán todas las imágenes nuevas
CARPETA_PRUEBAS = "../pruebas"

# Pedimos al usuario solo el nombre del archivo
nombre_archivo = input("Introduce el nombre del fichero con extensión (ej: imagen7.jpg): ")

# Construimos la ruta completa
ruta_prueba = os.path.join(CARPETA_PRUEBAS, nombre_archivo)

# Preprocesamos la imagen
img = preprocesar_imagen(ruta_prueba)

#Comprobamos si la función no devuelve None y devuelve un imagen válida
if img is not None:
    # Extraemos características HOG
    feat = extraer_hog(img)

    # Predicción del modelo
    pred = modelo.predict([feat])
    print("Predicción numérica:", pred[0])

    # Mostramos resultado
    if pred[0] == 0:
        print("La imagen es un GATO")
    else:
        print("La imagen es un PERRO")