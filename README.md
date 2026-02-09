# Práctica de clasificación binaria de perros y gatos usando HOG + SVM en Python.
## El proyecto es modular: preprocesamiento.py prepara imágenes, caracteristicas.py extrae HOG, modelo.py entrena SVM y main.py coordina todo.
## Las imágenes se organizan en carpetas por clase y se procesan automáticamente.
## Requisitos
NumPy: manejo de arrays y datos numéricos del dataset.
OpenCV: lectura y preprocesamiento de imágenes (grises, blur, bordes, redimensionado).
Scikit-Image: extracción de características visuales HOG.
Scikit-Learn: entrenamiento y predicción del clasificador SVM.
## Flujo del Pipeline
1. Lectura de imagen desde disco
2. Preprocesamiento:
 - Escala de grises
 - Redimensionado (64x64)
3. Extracción de características HOG
4. Construcción del dataset (X, y)
5. Entrenamiento del modelo SVM
6. Evaluación del modelo SVM
7. Predicción sobre nuevas imágenes