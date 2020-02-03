#!/usr/bin/env python
# coding: utf-8

# # Predicción Titanic

# 1. Importar Librerías
# 2. Importar los datos
# 3. Entender los datos
# 4. Pre-procesamiento o limpiado de datos
# 5. Aplicación de los algoritmos
# 6. Predicción utilizando los modelos

# 1. Importar Librerías
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from os import getcwd


# 2. Importar librerías de sklearn para los métodos
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# 2. Importar los datos
train_data = pd.read_csv('/Users/ramon/OneDrive/Master Big Data/Python/UD3_IntroAprendizajeAutomatico/titanic/train.csv', sep = ',')
test_data = pd.read_csv('/Users/ramon/OneDrive/Master Big Data/Python/UD3_IntroAprendizajeAutomatico/titanic/test.csv', sep = ',')



# 3. Entender los datos
print("Cantidad de datos del archivo de train: ",train_data.shape)
print("Cantidad de datos del archivo de test: ",test_data.shape)

print ("Datos que faltan en train: \n", pd.isnull(train_data).sum())
print ("\n\n Datos que faltan en test: \n\n", pd.isnull(test_data).sum())

# Tipos de datos de data_train
train_data.info()
print("\n\n")

# Tipos de datos de data_train
test_data.info()



print("Estadísticas del dataSet train: \n\n", train_data.describe())
print("\n\n Estadísticas del dataSet test: \n\n", test_data.describe())


#Calcular total mujeres
femaleData = train_data.loc[train_data['Sex']=='female']
femaleDF = pd.DataFrame(femaleData)
femaleTotal = femaleDF['Sex'].count()

#Calcular total hombres
maleData = train_data.loc[train_data['Sex']=='male']
maleDF = pd.DataFrame(maleData)
maleTotal = maleDF['Sex'].count()

# Mostrar datos
print("Total Mujeres: ", femaleTotal)
print("Total Hombres: ", maleTotal)


# Calcular mujeres que sobrevivieron
femaleSurvived = train_data.loc[train_data['Sex']=='female']['Survived'].sum()

# Calcular hombres que sobrevivieron
maleSurvived = train_data.loc[train_data['Sex']=='male']['Survived'].sum()


print("\nMujeres que sobrevivieron: ", femaleSurvived)
print("Hombre que sobrevivieron: ", maleSurvived)

print("\n% Mujeres que sobrevivieron: ", femaleSurvived/femaleTotal)
print("% Hombres que sobrevivieron: ", maleSurvived/maleTotal)


# 4. Limpiado de datos


# Convertir el tipo de datos de Sex de cadena/objeto a numérico 
train_data["Sex"].replace(['female','male'],[0,1],inplace=True)
test_data["Sex"].replace(['female','male'],[0,1],inplace=True)

# tras cambiar los datos, comentamos estas instrucciones para no volver a hacer el cambio,
# ya que dará error

# Calcular media de edad para poner esta en los datos que faltan
print(train_data["Age"].mean())
print(test_data["Age"].mean())

promedio = 30
train_data['Age'] = train_data['Age'].replace(np.nan, promedio)
test_data['Age'] = test_data['Age'].replace(np.nan, promedio)

pd.isnull(train_data).sum()

# Eliminar columna Cabin
train_data.drop(['Cabin'], axis =1, inplace=True)
test_data.drop(['Cabin'], axis =1, inplace=True)

pd.isnull(train_data).sum()

# Elimino las columnas que no considero necesarias
train_data = train_data.drop(['PassengerId','Name','Ticket','Embarked'], axis=1)
test_data =  test_data.drop(['Name','Ticket','Embarked'], axis=1)


# Eliminamos las filaes con datos perdidos, que como son muy pocas (2) las podemos eliminar por completo
train_data.dropna(axis=0, how='any', inplace=True)
test_data.dropna(axis=0, how='any', inplace=True)

train_data.head()

print('\n Datos que faltan en train: ',pd.isnull(train_data).sum())
print('\nDatos que faltan en test: ',pd.isnull(test_data).sum())

print('\n Tipos de datos de train: ',train_data.info())
print('\nTipos de datos de test: ',test_data.info())

print('\n Cabecera de train: ',train_data.head())
print('\nCabecera de test: ',test_data.head())


# 5. Aplicación de los algoritmos

# Separo la columna con la información de los sobrevivientes
X = np.array(train_data.drop(['Survived'], 1))
Y = np.array(train_data['Survived'])


# Separo los datos de train en entrenamiento y prueba para probar los algoritmos
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


## Regresión Logística
logreg = LogisticRegression()
logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_test)

print('Precisión Regresión Logística: ')
print(logreg.score(x_train,y_train))


## Support Vector Machine
# Definir método del algoritmo
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
print('Precisión Soporte de Vectores: ')
print(svc.score(x_train,y_train))



## K neighbors
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

print('Precisión K neighbors: ')
print(knn.score(x_train,y_train))




# 6. Predicción utilizando los modelos




## Support Vector Machine
prediccion_svc = svc.predict(test_data.drop('PassengerId',axis=1))
out_svc = pd.DataFrame({'PassengerId':ids,'Survived':prediccion_svc})
print('Precisión Regresión Logística: ')
print(out_svc.head())


# K neigbors

prediccion_knn = knn.predict(test_data.drop('PassengerId',axis=1))
out_knn = pd.DataFrame({'PassengerId':ids,'Survived':prediccion_knn})
print('Precisión Regresión Logística: ')
print(out_knn.head())

