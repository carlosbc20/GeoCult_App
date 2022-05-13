#!/usr/bin/env python
# coding: utf-8

# # VISUALIZACIÓN DE DATOS de GEOCULT-APP CLASIFICACIÓN CON NN
# <p> En este script se realizarán las siguientes tareas:</p>
# <p>En primer lugar, se leerán los datos de la base de datos local y se preprocesarán estos datos para visualizar gráficas que representen información sobre las actividades culturales en Madrid</p>
# <p>En segundo lugar, se implementará un algoritmo de clasificación, utilizando una red neuronal, que clasifique las actividades culturales de Madrid (almacenadas en una base de datos local), en actividades de pago o gratuitas considerando los siguientes atributos: "Audicence", "time", "event-location", "latitude" y "longitude".<p>
# <p>Script diseñado e implementado por: Carlos Breuer Carrasco y Emilio Delgado Muñoz.<p>

# # FASE 0 - LECTURA DE DOCUMENTOS DE LA BASE DE DATOS LOCAL

# <p>Conexión y lectura de los documentos almacenados en la base de datos local de MongoDB<p>

# In[1]:


"1. Importación de las librerías necesarias para ejecutar el script"
import requests 
import numpy as np 
import re
import os
import tarfile
import urllib
import pandas as pd
from pymongo import MongoClient
from os import listdir # library from the system used to read files from a directory
from os import path # library used to check the veracity of files and folders
import time # library used to control time spent on training operations
import matplotlib.pyplot as plt # library used to plot graphs and figures
import seaborn as sb # library used to make heatmaps from confusion matrices
from matplotlib import image # library used to import an image as a vector
import math # library used for math operations
import tensorflow as tf # machine learning library
from tensorflow import keras as k
from tensorflow.keras.utils import to_categorical # function from keras to make the one_hot matrix from the labels
from tensorflow.keras import backend as K # backend from keras
from sklearn.model_selection import train_test_split # used for splitting the data set into train and test set
from pandas.plotting import scatter_matrix
import collections
import seaborn as sns # import seaborn library to plot confusion matrix with porcentage values
from sklearn.cluster import DBSCAN


# In[2]:


"2. Definición de la función para realizar la conexión con la base de datos"
def get_db(CONNECTION_STRING):
    from pymongo import MongoClient
    import pymongo
    
    client = MongoClient(CONNECTION_STRING)
    return client  


# In[3]:


"3. Establecimiento de la conexión con MongoDB y acceso a la base de datos 'local'"
#Del cliente se selecciona la base de datos que se desea, en este caso la local
CONNECTION_STRING = 'mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false'
db = get_db(CONNECTION_STRING)['local']
print(db)


# In[4]:


"4. Acceso a la colección 'CultureEvents' de la base de datos 'local'. Se muestra información de la colección"
collection_name = db['CultureEvents']
print(collection_name)


# In[5]:


"5. Función auxiliar para convertir una hora de una actividad cultural a un entero, respetando y manteniendo la escala original"
def time_to_numerical(time_str):
    if (time_str == '00:00'):
        return 1
    elif (time_str == '07:00'):
        return 29
    elif (time_str == '07:15'):
        return 30
    elif (time_str == '07:30'):
        return 31
    elif (time_str == '07:45'):
        return 32
    elif (time_str == '08:00'):
        return 33
    elif (time_str == '08:15'):
        return 34
    elif (time_str == '08:30'):
        return 35
    elif (time_str == '08:45'):
        return 36
    elif (time_str == '09:00'):
        return 37
    elif (time_str == '09:15'):
        return 38
    elif (time_str == '09:30'):
        return 39
    elif (time_str == '09:45'):
        return 40
    elif (time_str == '10:00'):
        return 41
    elif (time_str == '10:15'):
        return 42
    elif (time_str == '10:30'):
        return 43
    elif (time_str == '10:45'):
        return 44
    elif (time_str == '11:00'):
        return 45
    elif (time_str == '11:15'):
        return 46
    elif (time_str == '11:30'):
        return 47
    elif (time_str == '11:45'):
        return 48
    elif (time_str == '12:00'):
        return 49
    elif (time_str == '12:15'):
        return 50
    elif (time_str == '12:30'):
        return 51
    elif (time_str == '12:45'):
        return 52
    elif (time_str == '13:00'):
        return 53
    elif (time_str == '13:15'):
        return 54
    elif (time_str == '13:30'):
        return 55
    elif (time_str == '13:45'):
        return 56
    elif (time_str == '14:00'):
        return 57
    elif (time_str == '14:15'):
        return 58
    elif (time_str == '14:30'):
        return 59
    elif (time_str == '14:45'):
        return 60
    elif (time_str == '15:00'):
        return 61
    elif (time_str == '15:15'):
        return 62
    elif (time_str == '15:30'):
        return 63
    elif (time_str == '15:45'):
        return 64
    elif (time_str == '16:00'):
        return 65
    elif (time_str == '16:15'):
        return 66
    elif (time_str == '16:30'):
        return 67
    elif (time_str == '16:45'):
        return 68
    elif (time_str == '17:00'):
        return 69
    elif (time_str == '17:15'):
        return 70
    elif (time_str == '17:30'):
        return 71
    elif (time_str == '17:45'):
        return 72
    elif (time_str == '18:00'):
        return 73
    elif (time_str == '18:15'):
        return 74
    elif (time_str == '18:30'):
        return 75
    elif (time_str == '18:45'):
        return 76
    elif (time_str == '19:00'):
        return 77
    elif (time_str == '19:15'):
        return 78
    elif (time_str == '19:30'):
        return 79
    elif (time_str == '19:45'):
        return 80
    elif (time_str == '20:00'):
        return 81
    elif (time_str == '20:15'):
        return 82
    elif (time_str == '20:30'):
        return 83
    elif (time_str == '20:45'):
        return 84
    elif (time_str == '21:00'):
        return 85
    elif (time_str == '21:15'):
        return 86
    elif (time_str == '21:30'):
        return 87
    elif (time_str == '21:45'):
        return 88
    elif (time_str == '22:00'):
        return 89
    elif (time_str == '22:15'):
        return 90
    elif (time_str == '22:30'):
        return 91
    elif (time_str == '22:45'):
        return 92
    elif (time_str == '23:00'):
        return 93
    else:
        return 0


# In[6]:


"6. Lectura de los datos de la colección de la base de datos a un dataframe de la librería pandas"
def read_mongo(db, collection, query={}):
    """ Read from Mongo and Store into DataFrame """

    # Make a query to the specific DB and Collection
    cursor = db[collection].find(query)
    
    # Define the dataframe
    df = pd.DataFrame(columns=("id","title","free","price", "dtstart", "dtend", "time", "time_code", "audience", "event-location", "latitude", "longitude"))

    # Expand the cursor and construct the DataFrame
    contador = 0
    list_words_prices = ['euros', 'Gratuito', 'gratuita', 'Gratis', 'libre', '']
    list_hours_time = ['19:01', '18:40']
    for doc in cursor:
        contador = contador + 1
        if(any(word in doc['price'] for word in list_words_prices) and doc['time'] not in list_hours_time):
            if ('euros' in doc['price']):
                price_list = [int(s) for s in re.findall(r'\b\d+\b', doc['price'])]
                precio = price_list[0]
            else:
                precio = 0
            if doc['time'] == '':
                hora_doc = "Sin especificar"
            else:
                hora_doc = doc['time']
            numerical_time = time_to_numerical(doc['time'])
            df = df.append({'id' : doc['id'],'title':doc['title'], 'free' : float(doc['free']), 'price' : float(precio),
                            'dtstart':doc['dtstart'], 'dtend':doc['dtend'], 'time':hora_doc, 'time_code':float(numerical_time),
                            'audience': doc['audience'], 'event-location' : doc['event-location'],
                            'latitude' : doc['location']['latitude'], 'longitude' : doc['location']['longitude']},
                           ignore_index=True, verify_integrity=False, sort=None) 
    
    df.drop_duplicates()
    #df = df.drop(df[(df.price > 50.0)].index)
    print("Documentos devueltos: ", df.shape[0])
    return df


# In[7]:


geoCult_data = None
geoCult_data = read_mongo(db, 'CultureEvents', query={})
geoCult_data.head()


# In[8]:


geoCult_data["audience"].value_counts()


# In[9]:


from sklearn.preprocessing import OrdinalEncoder

ord_enc = OrdinalEncoder()
geoCult_data["audience_code"] = ord_enc.fit_transform(geoCult_data[["audience"]])
geoCult_data["audience_code"] = geoCult_data["audience_code"].astype('float64')
geoCult_data[["audience", "audience_code"]].head(11)


# In[10]:


geoCult_data["audience_code"].value_counts()


# In[11]:


geoCult_data["event-location"].value_counts()


# In[12]:


geoCult_data["event-location_code"] = ord_enc.fit_transform(geoCult_data[["event-location"]])
geoCult_data["event-location_code"] = geoCult_data["event-location_code"].astype('float64')
geoCult_data[["event-location", "event-location_code"]].head(11)


# In[13]:


geoCult_data["event-location_code"].value_counts()


# In[14]:


geoCult_data["price"].value_counts()


# In[15]:


geoCult_data["free"].value_counts()


# In[16]:


geoCult_data.info()


# In[17]:


geoCult_data.head(10)


# In[18]:


geoCult_data.hist(bins=50, figsize=(20,15))
plt.show()


# In[53]:


attributes = ["latitude", "longitude", "price", "event-location_code", "free", "time_code", "audience_code"]
scatter_matrix(geoCult_data[attributes], figsize=(12, 8))


# In[19]:


geoCult_data.describe()


# In[20]:


train_set, test_set = train_test_split(geoCult_data, test_size=0.2, random_state=42)

print("train = ", len(train_set), "test = ", len(test_set))


# In[21]:


"Se eliminan las columnas innecesarias que no se utilizarán para entrenar la red neuronal"
geoCult_data.pop('id')
geoCult_data.pop('title')
geoCult_data.pop('dtstart')
geoCult_data.pop('dtend')
geoCult_data.pop('event-location')
geoCult_data.pop('audience')
geoCult_data.pop('time')
geoCult_data.info()


# In[22]:


geoCult_data.info()


# # ALGORITMO DBScan

# In[71]:


clustering = DBSCAN(eps=0.003, min_samples=2).fit(geoCult_data)
clustering.labels_

clustering


# In[72]:


print(clustering.labels_)
print(collections.Counter(clustering.labels_))


# In[73]:


plt.scatter(geoCult_data['latitude'], geoCult_data['event-location_code'], c=clustering.labels_)


# In[74]:


plt.scatter(geoCult_data['longitude'], geoCult_data['latitude'], c=clustering.labels_)


# In[75]:


plt.scatter(geoCult_data['time_code'], geoCult_data['latitude'], c=clustering.labels_)


# In[76]:


plt.scatter(geoCult_data['longitude'], geoCult_data['price'], c=clustering.labels_)


# In[77]:


geoCult_data['labels'] = clustering.labels_


# In[78]:


geoCult_data


# In[79]:


geoCult_data_no_outliers = geoCult_data[(geoCult_data['labels'] != -1)]


# In[80]:


geoCult_data_no_outliers


# In[81]:


plt.scatter(geoCult_data_no_outliers['latitude'], geoCult_data_no_outliers['event-location_code'], c=geoCult_data_no_outliers['labels'])


# In[82]:


plt.scatter(geoCult_data_no_outliers['longitude'], geoCult_data_no_outliers['latitude'], c=geoCult_data_no_outliers['labels'])


# In[51]:


plt.scatter(geoCult_data_no_outliers['time_code'], geoCult_data_no_outliers['latitude'], c=geoCult_data_no_outliers['labels'])


# In[83]:


plt.scatter(geoCult_data_no_outliers['longitude'], geoCult_data_no_outliers['price'], c=geoCult_data_no_outliers['labels'])

