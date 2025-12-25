#Modelo aplicado a Iris,
#Se busca un mejor modelo y se encuentra a LDA como mejor.
#No se usa grid por no considerarse encesario o relevantes en los hiperparametros de LDA.

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

#Carga de datos
DIRECCION = (r'F:\Análisis Profundo - FAT32\G-TALENT\MACHINE LEARNING\\')
LIBRO = "iris.xlsx"
LIBRO_2 = "iris_nuevos_datos.xlsx"

df = pd.read_excel(DIRECCION + LIBRO)
df.head(5)

#Seleccción de columnas.
X = df.drop('clase', axis=1).values
y = df['clase'].values

#Train test split.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y) #42 por cenvención, puede afectar el score.

#Buscamos un mejor modelo de entre varios acordes al problema.
#Importamos todas las librerías
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC #

#Se guardan los modelos en un array.
models=[]
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('ART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVC', SVC()))

#Se guaran resultados en un array.
results=[]
names=[]
for name, model in models:
  #kfold =model_selection.KFold(n_splits=10, shuffle=True, random_state=42)
  kfold = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
  cv_results = model_selection.cross_val_score(model, X, y, cv=kfold)
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  
  print(msg)
  msg
  
#En este caso el mejor modelo fué LDA. No usa grid por no aportar demasiado en este caso. 
LDA_final = LinearDiscriminantAnalysis()
#Entrenamiento. Se hace un entrenamiento simple a final de cuentas.
LDA_final.fit(X_train, y_train)

#cCarga de libro origen de datos a predecir.
df_predict = pd.read_excel(DIRECCION + LIBRO_2)
X_predict = df_predict.values

#Entrenamiento
y_predict = LDA_final.predict(X_predict)
y_predict
