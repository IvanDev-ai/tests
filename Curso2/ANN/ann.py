import numpy as np
import matplotlib as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import keras
from keras.models import Sequential
from keras.layers import Dense


#Recoger dataset
dataset = pd.read_csv('Churn_Modelling.csv')

#Dividir dataset entre variables de entrada y variable dependiente
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:13].values

#Estandarizar las variables de entrada "Spain" -> "1"
labelencoder_X1 = LabelEncoder()
X[:1] = labelencoder_X1.fit_transform(X[:1])
labelencoder_X2 = LabelEncoder()
X[:2] = labelencoder_X2.fit_transform(X[:2])

#Evitar multicolinealidad o trampa de variables ficticias
transformer = ColumnTransformer(transformers=[("Churn_Modelling",OneHotEncoder(categories='auto'),[1])], remainder='passthrough')
X = transformer.fit_transform(X)
X = X[:, 1:]

#Dividir dataset preprocesado  en conjunto de entrenamiento y conjunto de testing
X_train, X_test, Y_train,  Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#Estandarizar variables X para que sean valores cercanos a 0
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

classifier = Sequential()