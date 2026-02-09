from sklearn.metrics import accuracy_score, classification_report

def evaluar_modelo(modelo, X_test, y_test):
 """
 Evalúa el modelo con datos de prueba.
 """
 y_pred = modelo.predict(X_test)

 accuracy = accuracy_score(y_test, y_pred) #Calcula la precisión del modelo
 print(f"Precisión del modelo: {accuracy:.2f}")
 
 print("\nInforme de clasificación:")
 print(classification_report(y_test, y_pred))