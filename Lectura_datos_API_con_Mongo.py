#!/usr/bin/env python
# coding: utf-8

# <h1>LECTURA O POBLACIÓN DE DATOS DE API A MONGODB</h1>
# <p>Autores: CARLOS BREUER & EMILIO DELGADO (USO DE PYMONGO)</p>
# <p>API utilizada: <a href="https://datos.madrid.es/portal/site/egob/menuitem.214413fe61bdd68a53318ba0a8a409a0/?vgnextoid=b07e0f7c5ff9e510VgnVCM1000008a4a900aRCRD&vgnextchannel=b07e0f7c5ff9e510VgnVCM1000008a4a900aRCRD&vgnextfmt=default">Portal de datos libres de Madrid</a></p>
# <p>Cabe destacar que antes de ejecutar este script será necesario haber creado la colección CultureEvents sobre la base de datos 'Local' de MongoDB y haber creado un índice secundario y otro geoespacial de la siguiente manera:</p>
# <p>db.CultureEvents.createIndex({"id":1},{ unique: true } );</p>
# <p>db.CultureEvents.createIndex({'location':"2dsphere"});</p>

# # LECTURA DE DATOS DE LA API

# In[1]:


"1. Importación de librerías necesarias"
import requests 
import numpy as np 
import pandas as pd
from pymongo.errors import BulkWriteError


# In[2]:


"2. Petición de tipo GET desde la capa REST"
r = requests.get('https://datos.madrid.es/egob/catalogo/206974-0-agenda-eventos-culturales-100.json')


# In[3]:


"3. Obtención del contenido de la petición realizada en forma de diccionario (JSON) y se muestra su contenido"
my_dict = r.json()
my_dict


# In[4]:


"4. Transformación del diccionario en un dataframe de Pandas."
df = pd.DataFrame(my_dict['@graph'])


# In[5]:


"5. Se realiza una copia del dataframe para procesar los datos sobre la copia en lugar de sobre el dataframe original"
df_copy = df.copy()


# In[6]:


"6. Se muestra información sobre las columnas del dataframe"
df_copy.info()


# # FASE DE PRE-PROCESADO

# In[7]:


"7. Los valores nulos de la columna AUDIENCE se sustituyen por la cadena 'No audience'"
df_copy['audience'].fillna('No audience',inplace=True)
"8. Los valores nulos de la columna @Type se sustituyen por una serie del tipo 'object'"
df_copy['@type'].fillna(pd.Series(dtype=object), inplace=True)
"9. Se eliminan todas las tuplas o filas cuyos valores en el campo 'location' sean nulos, es decir, no existe ubicación"
df_clean = df_copy[(df_copy['location'].isna()==False)]


# In[8]:


"10. Se vuelve a mostrar información sobre las columnas del dataframe, esta vez ya pre-procesado"
df_clean.info()


# In[9]:


"11. Se muestran las primeras 20 filas del dataframe pre-procesado"
df_clean.head(20)


# # INSERCIÓN DE LOS DOCUMENTOS LEÍDOS DE LA API

# In[10]:


"12. Definición de la función para realizar la conexión con la base de datos"
def get_db(CONNECTION_STRING):
    from pymongo import MongoClient
    import pymongo
    
    client = MongoClient(CONNECTION_STRING)
    return client    


# In[11]:


"13. Establecimiento de la conexión con MongoDB y acceso a la base de datos 'local'"
#Del cliente se selecciona la base de datos que se desea, en este caso la local
CONNECTION_STRING = 'mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false'
db = get_db(CONNECTION_STRING)['local']
print(db)


# In[12]:


"14. Acceso a la colección 'CultureEvents' de la base de datos 'local'. Se muestra información de la colección"
collection_name = db['CultureEvents']
print(collection_name)


# In[13]:


"15. Se insertan las filas sin duplicados de aquellos nuevos eventos culturales que se hayan recuperado de la API y preprocesado previamente"
try:
    db.CultureEvents.insert_many(df_clean.to_dict('records'), ordered = False)
except BulkWriteError:
    pass

