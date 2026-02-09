import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def entrenar_modelo(X, y):
 """
 Separa los datos y entrena un clasificador SVM.
 """
 X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.3, random_state=42
 )

 modelo = SVC(kernel='linear')
 modelo.fit(X_train, y_train)
 
 return modelo, X_test, y_test