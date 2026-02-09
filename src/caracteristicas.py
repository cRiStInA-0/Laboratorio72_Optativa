from skimage.feature import hog

def extraer_hog(imagen):
 """
 Extrae características HOG a partir de una imagen procesada.
 """

 caracteristicas = hog(imagen,
 pixels_per_cell=(4, 4), #tamaño de cada celda, 8x8
 cells_per_block=(2, 2), #cantidad de celdas que forman un bloque
 transform_sqrt=True, #ayuda con las variaciones de iluminación
 feature_vector=True #devuelve un vector plano listo para ML
 )

 #Devuelve un array NumPy que representa la imagen en términos de HOG.
 return caracteristicas 