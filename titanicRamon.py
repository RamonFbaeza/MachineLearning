#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


# 2. Importar librerías de sklearn para los métodos
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# In[4]:


# 2. Importar los datos
train_data = pd.read_csv('/Users/ramon/OneDrive/Master Big Data/Python/UD3_IntroAprendizajeAutomatico/titanic/train.csv', sep = ',')
test_data = pd.read_csv('/Users/ramon/OneDrive/Master Big Data/Python/UD3_IntroAprendizajeAutomatico/titanic/test.csv', sep = ',')


# In[5]:


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


# In[6]:


print("Estadísticas del dataSet train: \n\n", train_data.describe())
print("\n\n Estadísticas del dataSet test: \n\n", test_data.describe())


# In[7]:


#Convierto los datos de sexos en números
# Cuidado que si vuelvo a ejecutar esto me da error al ya haberlo convertido
train_data['Sex'].replace(['female','male'],[0,1],inplace=True)
test_data['Sex'].replace(['female','male'],[0,1],inplace=True)


# In[8]:


train_data.info()


# In[9]:


# ASI NO FUNCIONA - train_data.select('Embarked').distinct()

# ASI SI FUNCIONA - Muestra todos los valores distintos de la columna embarked
train_data["Embarked"].unique()


# In[10]:


#Combierto los valores de Embarked a número
train_data['Embarked'].replace(['Q','S', 'C'],[0,1,2],inplace=True)
test_data['Embarked'].replace(['Q','S', 'C'],[0,1,2],inplace=True)


# In[11]:


pd.isnull(train_data).sum()


# In[12]:


## Valor medo de 'Age'


# In[13]:


promedio = 30

# Reemplazo los valores que no existen  (nan) por el promedio calculado anteriormente
train_data['Age'] = train_data['Age'].replace(np.nan, promedio)
test_data['Age'] = test_data['Age'].replace(np.nan, promedio)


# In[14]:


## Ya no hay valores perdidos en Age
pd.isnull(train_data).sum()


# In[15]:


# Elimino la columna 'Cabin' ya que tiene muchos datos perdidos
train_data.drop(['Cabin'], axis = 1, inplace=True)
test_data.drop(['Cabin'], axis = 1, inplace=True)


# In[16]:


pd.isnull(train_data).sum()


# In[17]:


# Falta eliminar las dos filas de que tienen datos perdidos de 'Embarked' y que son únicamente dos
# así que no hace falta reeemplazarlos

train_data.dropna(axis=0, how='any', inplace=True)
test_data.dropna(axis=0, how='any', inplace=True)


# In[18]:


print(train_data.head())
print(test_data.head())


# In[19]:


## Elimino las columnas que no considero necesarias
train_data = train_data.drop(['PassengerId','Name','Ticket'], axis=1)
test_data = test_data.drop(['PassengerId','Name','Ticket'], axis=1)


# In[21]:


print(train_data.head())
print(test_data.head())


# In[22]:


## Uso de Algoritmos

#Separo la columna con la información de los sobrevivientes
X = np.array(train_data.drop(['Survived'], 1))
y = np.array(train_data['Survived'])


#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


##Regresión logística
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
Y_pred = logreg.predict(X_test)
print('Precisión Regresión Logística:')
print(logreg.score(X_train, y_train))


# In[23]:


##Support Vector Machines
svc = SVC()
svc.fit(X_train, y_train)
Y_pred = svc.predict(X_test)
print('Precisión Soporte de Vectores:')
print(svc.score(X_train, y_train))


# In[24]:


##K neighbors
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)
print('Precisión Vecinos más Cercanos:')
print(knn.score(X_train, y_train))

