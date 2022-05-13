#!/usr/bin/env python
# coding: utf-8

# <h1>APLICACIÓN GEOCULT-ADRID A MONGODB</h1>
# <p>Autores: CARLOS BREUER & EMILIO DELGADO (USO DE PYMONGO)</p>
# <p>API utilizada: <a href="https://datos.madrid.es/portal/site/egob/menuitem.214413fe61bdd68a53318ba0a8a409a0/?vgnextoid=b07e0f7c5ff9e510VgnVCM1000008a4a900aRCRD&vgnextchannel=b07e0f7c5ff9e510VgnVCM1000008a4a900aRCRD&vgnextfmt=default">Portal de datos libres de Madrid</a></p>
# <p>Cabe destacar que antes de ejecutar este script será necesario haber creado la colección CultureEvents sobre la base de datos 'Local' de MongoDB y haber creado un índice secundario y otro geoespacial de la siguiente manera:</p>
# <p>db.CultureEvents.createIndex({"id":1},{ unique: true } );</p>
# <p>db.CultureEvents.createIndex({'location':"2dsphere"});</p>

# # CONEXIÓN CON LA COLECCIÓN DE LA BASE DE DATOS

# In[1]:


"1. Importación de librerías necesarias"
import requests 
import numpy as np 
import pandas as pd
from pymongo.errors import BulkWriteError
from geopy.geocoders import Nominatim 
from pprint import pprint
from IPython.display import display
import folium
from datetime import datetime
import math
from tkinter import *
from tkinter import ttk
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import io
import sys
import random
from PyQt5 import QtWidgets, QtWebEngineWidgets
from IPython.display import clear_output as clear
from PIL import Image, ImageTk


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


# # ACCESOS A LA COLECCIÓN DE LA BD DE EJEMPLO

# In[5]:


"5. Se realiza un acceso de un documento de ejemplo de la colección"
doc = None
for i in db.CultureEvents.find({'id':'11630153'}):
    doc=i
doc['location']


# In[6]:


"6. Se realiza un acceso de un documento de la colección mediante el título mediante la librería geopy"
loc = Nominatim(user_agent="GetLoc") 
  
getLoc = loc.geocode("Centro Cultural Casa de Vacas") 
  
print(getLoc.address) 
  
print("Latitude = ", getLoc.latitude, "\n") 
print("Longitude = ", getLoc.longitude) 


# # DEFINICIÓN DE FUNCIONES

# In[7]:


"7. Función para mostrar ubicaciones de actividades culturales en el mapa según varios modos"
"   Descripción de modos de la función"
"   MODO 1: Muestra en el mapa  las próximas actividades culturales en Madrid"
"   MODO 2: Muestra en el mapa las próximas actividades culturales en Madrid a 1km de la ubicación del usuario"
"   MODO 7: Muestra en el mapa todas las actividades culturales sucedidas en Madrid almacenadas en la base de datos"
"   Retorna dos cadena con el nº de ubicaciones de actividades consultadas y con el nº de actividades descartadas"
def showMap(latitude, longitude, minDistance, maxDistance, mode, result):
    clear()
    #Create the map
    cursor = db.CultureEvents.find({'location':{'$near':{'$geometry': {'type':'Point', 'coordinates':[latitude,longitude]}, '$minDistance':minDistance,'$maxDistance':maxDistance}}})
    numTotal = 0
    cont_act_descartadas = 0
    
    if mode == 7:
        index = 0
        num_docs_cursor = len(list(cursor.clone()))
        #print("TOTAL: ", num_docs_cursor)
        activities_start_pos = random.randint(0, num_docs_cursor-500)
        activities_end_pos = activities_start_pos + 500
        #print("ACTIVITIES START POSITION: ", activities_start_pos)
        #print("ACTIVITIES END POSITION: ", activities_end_pos)
    map = None
    
    for doc in cursor:
        if(numTotal == 0):
            # El estilo se puede cambiar a 'Stamen Toner' o 'Stamen Terrain'
            map = folium.Map(location=[doc['location']['latitude'], doc['location']['longitude']], tiles="Stamen Terrain", zoom_start=12)
            if(mode == 2): # Se muestra la ubicación del usuario
                folium.Marker([latitude, longitude], icon=folium.Icon(icon='home', color='orange',icon_color='#ffffff'),
                              popup="MI UBICACIÓN\nLatitud: "+str(latitude)+"\nLongitud: "+str(longitude)).add_to(map)    

        # Se obtiene la fecha actual
        today = datetime.today()
        # Se convierte la fecha a datetime
        fecha_act = doc['dtstart']
        if(type(fecha_act) == str):
            fecha_act_formated = fecha_act[0:len(fecha_act)-2]
            datetime_act = datetime.strptime(fecha_act_formated, "%Y-%m-%d %H:%M:%S")
            
            if(mode == 1 and today < datetime_act):
                folium.Marker([doc['location']['latitude'], doc['location']['longitude']], popup=doc['event-location']).add_to(map)    
                numTotal = numTotal + 1
            elif(mode == 2 and today < datetime_act):
                folium.Marker([doc['location']['latitude'], doc['location']['longitude']], popup="Título: "+doc['title']+", Localización: "+doc['event-location']+", Fecha inicio: "+doc['dtstart']).add_to(map)    
                numTotal = numTotal + 1
            elif (mode == 7):
                if index > activities_end_pos:
                    break
                elif (index >= activities_start_pos and index < activities_end_pos):
                    folium.Marker([doc['location']['latitude'], doc['location']['longitude']], popup=doc['event-location']).add_to(map)    
                    numTotal = numTotal + 1
                index = index + 1
        else:
            cont_act_descartadas = cont_act_descartadas + 1
    
    print("Número de actividades culturales consultadas: ",numTotal)
    print("Número de actividades culturales descartadas por formato inválido: ",cont_act_descartadas)
    result.append("Número de actividades culturales consultadas: "+str(numTotal))
    result.append("Número de actividades culturales descartadas por formato inválido: "+str(cont_act_descartadas))
    display(map)


# In[8]:


"8. Función para mostrar la actividad cuyo título coincide con la cadena de entrada"
"   Retorna una cadena con el nº de ubicaciones de actividades consultadas"
def showMap_Activity_Title(title_value, result):
    clear()
    cont = 0
    for doc in db.CultureEvents.find({'title':title_value}):
        if(cont == 0):
            # El estilo se puede cambiar a 'Stamen Toner' o 'Stamen Terrain'
            map = folium.Map(location=[doc['location']['latitude'],  doc['location']['longitude']], tiles="Stamen Terrain", zoom_start=12)
        folium.Marker([doc['location']['latitude'], doc['location']['longitude']], popup="Título: "+doc['title']+", Localización: "+doc['event-location']+", Fecha inicio: "+doc['dtstart']+", Precio: "+doc['price']).add_to(map)
        cont = cont + 1
        
    if (cont >= 1):
        print("Número de actividades culturales consultadas: ", cont)
        result.append("Número de actividades culturales consultadas: "+str(cont))
        display(map)


# In[9]:


"9. Función para mostrar las actividades cuya fecha de inicio coincide con la fecha especifica como parámetro"
"   Retorna una cadena con el nº de ubicaciones de actividades consultadas"
def showMap_Activities_Date(date_value, result):
    clear()
    cont = 0
    date_value_formated = date_value + ".0"
    
    for doc in db.CultureEvents.find({'dtstart':date_value_formated}):
        if(cont == 0):
            # El estilo se puede cambiar a 'Stamen Toner' o 'Stamen Terrain'
            map = folium.Map(location=[doc['location']['latitude'],  doc['location']['longitude']], tiles="Stamen Terrain", zoom_start=12)
        folium.Marker([doc['location']['latitude'], doc['location']['longitude']], popup="Título: "+doc['title']+", Localización: "+doc['event-location']+", Fecha inicio: "+doc['dtstart']).add_to(map)
        cont = cont + 1
            
    if (cont >= 1):
        print("Número de actividades culturales consultadas: ", cont)
        result.append("Número de actividades culturales consultadas: "+str(cont))
        display(map)


# In[10]:


"10. Función para mostrar las actividades que son gratuitas o de pago dependiendo del valor del parámetro de entrada"
"   Si el parámetro de entrada es 0 se mostrarán en el mapa las actividades de pago."
"   Si el parámetro de entrada es 1 se mostrarán en el mapa las actividades gratuitas."
"   Retorna dos cadena con el nº de ubicaciones de actividades consultadas y con el nº de actividades descartadas"
def showMap_Free_Or_Not_FreeActivities(free_value, result):
    clear()
    cont = 0
    cont_act_descartadas = 0
    # Se obtiene la fecha actual
    today = datetime.today()
    
    for doc in db.CultureEvents.find({'free':free_value}):
        # Se convierte la fecha a datetime
        fecha_act = doc['dtstart']
        if(type(fecha_act) == str):
            fecha_act_formated = fecha_act[0:len(fecha_act)-2]
            datetime_act = datetime.strptime(fecha_act_formated, "%Y-%m-%d %H:%M:%S")
            
            if(today < datetime_act):
                if(cont == 0):
                    # El estilo se puede cambiar a 'Stamen Toner' o 'Stamen Terrain'
                    map = folium.Map(location=[doc['location']['latitude'],  doc['location']['longitude']], tiles="Stamen Terrain", zoom_start=12)
                folium.Marker([doc['location']['latitude'], doc['location']['longitude']], popup="Localización: "+doc['event-location']+", Fecha inicio: "+doc['dtstart']).add_to(map)
                cont = cont + 1
        else:
            #print("El formato de la fecha recibida no es el correcto")
            cont_act_descartadas = cont_act_descartadas + 1
    
    if (cont >= 1):
        print("Número de actividades culturales consultadas: ", cont)
        print("Número de actividades culturales descartadas por formato de fecha no contemplado: ", cont_act_descartadas)
        result.append("Número de actividades culturales consultadas: "+str(cont))
        result.append("Número de actividades culturales descartadas por formato de fecha no contemplado: "+str(cont_act_descartadas))
        display(map)


# <h2>MENÚ PRINCIPAL DE LA APLICACIÓN GRÁFICA CON TKINTER</h2>

# In[ ]:


"12. MENÚ PRINCIPAL DE LA APLICACIÓN GRÁFICA CON TKINTER"

print("APLICACIÓN GEOCULT-MADRID SOBRE ACTIVIDADES CULTURALES")

# tkinter GUI
root= tk.Tk()
root.title('GeoCult-Madrid - Graphic Tkinter Interface')
root.geometry('950x480')
root.config(bg='#b29bff')

canvas1 = tk.Canvas(root, bd = 8, width = 890, height = 440)
canvas1.pack()

rectangles = []

canvas1.create_rectangle(285, 60, 870, 160)
canvas1.create_rectangle(600, 168, 850, 285)
canvas1.create_rectangle(295, 295, 890, 370)

# Título de la aplicación
label_title = tk.Label(root, text="GeoCult-Madrid (Aplicación sobre actividades culturales en Madrid)",
                       justify = 'center', font=("Arial", 20))
canvas1.create_window(450, 30, window=label_title)

result = []

# Coordenadas de ubicación del usuario
label_title = tk.Label(root, text="Coordenadas de la ubicación del usuario", justify = 'center', font=("Arial", 13))
canvas1.create_window(438, 80, window=label_title)

# Latitud
label_latitud = tk.Label(root, text="Latitud:", justify = 'center', font=("Arial", 13))
canvas1.create_window(335, 109, window=label_latitud)
entry_latitud = tk.Entry (root) # create entry box for latitud
canvas1.create_window(450, 109, window=entry_latitud)

# Longitud
label_longitud = tk.Label(root, text="Longitud:", justify = 'center', font=("Arial", 13))
canvas1.create_window(340, 139, window=label_longitud)
entry_longitud = tk.Entry (root) # create entry box for longitud
canvas1.create_window(450, 139, window=entry_longitud)

# Radio de proximidad de la actividad
label_radio = tk.Label(root, text="| Radio de proximidad (km):", justify = 'center', font=("Arial", 13))
canvas1.create_window(700, 80, window=label_radio)
entry_radio = tk.Entry (root, width=7) # create entry box for radio de proximidad
canvas1.create_window(830, 80, window=entry_radio)

# Título de la actividad
label_titulo = tk.Label(root, text="Título:", justify = 'center', font=("Arial", 13))
canvas1.create_window(664, 267, window=label_titulo)
entry_titulo = tk.Entry (root) # create entry box for titulo
canvas1.create_window(755, 267, window=entry_titulo)

# Fecha en el formato 'YYYY-MM-DD HH:mm:SS'
label_fecha = tk.Label(root, text="Fecha:", justify = 'center', font=("Arial", 13))
canvas1.create_window(660, 314, window=label_fecha)
label_formato = tk.Label(root, text="(formato fecha: 'YYYY-MM-DD HH:mm:SS')", justify = 'center', font=("Arial", 12))
canvas1.create_window(732, 344, window=label_formato)
entry_fecha = tk.Entry (root) # create entry box for fecha
canvas1.create_window(755, 314, window=entry_fecha)

    
# Función para mostrar, bien las próximas actividades culturales en Madrid,
# o bien todas las actividades culturales de Madrid existentes en la base de datos
def next_global_points(opcion):
    if(opcion == 1):
        showMap(40.4133227,-3.7085303, 0, 500000, 1, result)
    elif(opcion == 7):
        showMap(40.4133227,-3.7085303, 0, 500000, 7, result)
    mostrarResultadosConsulta2()
    
# Función para obtener las coordenadas geográficas y obtener el mapa solicitado
def location_values():
    Latitud = None # 1st input variable: latitud
    Latitud = float(entry_latitud.get()) 
    
    Longitud = None # 2nd input variable: longitud
    Longitud = float(entry_longitud.get())
    
    Radio = None
    Radio = float(entry_radio.get())
    print("Latitud: ", Latitud)
    print("Longitud: ", Longitud)
    print("Radio: ", Radio)
    
    showMap(Latitud, Longitud, 0, Radio * 1000, 2, result)
    mostrarResultadosConsulta2()
    
# Función para obtener la fecha introducida y obtener el mapa solicitado
def date_values():
    Date = None # 1st input variable: date
    Date = entry_fecha.get()
    
    showMap_Activities_Date(Date, result)
    mostrarResultadosConsulta1()
    
# Función para obtener el título introducido y obtener el mapa de actividades solicitado
def title_values():
    Title = None # 1st input variable: date
    Title = entry_titulo.get()
    
    showMap_Activity_Title(Title, result)
    mostrarResultadosConsulta1()
    
# Función para mostrar, bien las próximas actividades culturales en Madrid gratuitas,
# o bien las próximas actividades culturales de Madrid de pago
def free_or_not_free_activities(free):
    if(free == 1):
        showMap_Free_Or_Not_FreeActivities(free, result)
    elif(free == 0):
        showMap_Free_Or_Not_FreeActivities(free, result)
    mostrarResultadosConsulta2()
    
# Función para mostrar en la interfaz gráfica 2 de los resultados obtenidos
def mostrarResultadosConsulta2():
    label_text_result1 = tk.Label(root, text=result[0], bg='yellow', font=("Arial", 13))
    canvas1.create_window(430, 400, window=label_text_result1)
    label_text_result2 = tk.Label(root, text=result[1], bg='yellow', font=("Arial", 13))
    canvas1.create_window(430, 430, window=label_text_result2)
    if(len(result) > 0):
        result.clear()
    label_text_result1.after(6000 , lambda: label_text_result1.destroy())
    label_text_result2.after(6000 , lambda: label_text_result2.destroy())
    
# Función para mostrar en la interfaz gráfica 1 de los resultados obtenidos
def mostrarResultadosConsulta1():
    label_text_result1a = tk.Label(root, text=result[0], bg='yellow', font=("Arial", 13))
    canvas1.create_window(430, 400, window=label_text_result1a)
    if(len(result) > 0):
        result.clear()
    label_text_result1a.after(6000 , lambda: label_text_result1a.destroy())


button1 = tk.Button (root, text='Próximas actividades\nculturales en Madrid',
                     command= lambda: next_global_points(1), bg='#abffff',
                     height = 3, width = 24, font=("Arial", 13)) # button to call the 'values' command above 
canvas1.create_window(165, 110, window=button1)


button2 = tk.Button (root, text='Próximas actividades más cercanas a mi\nubicación a un radio de distancia (km)',
                     command = location_values, bg='#ffd565',
                     height = 2, width = 34, font=("Arial", 13)) # button to call the 'values' command above 
canvas1.create_window(695, 125, window=button2)


button3 = tk.Button (root, text='Buscar actividades culturales\nque suceden en una fecha',
                     command=date_values, bg='#f3ff98',
                     height = 2, width = 26, font=("Arial", 13)) # button to call the 'values' command above 
canvas1.create_window(435, 330, window=button3)


button4 = tk.Button (root, text='Buscar actividad cultural\npor el título',
                     command=title_values, bg='#d998ff',
                     height = 3, width = 24, font=("Arial", 13)) # button to call the 'values' command above 
canvas1.create_window(725, 210, window=button4)


button5 = tk.Button (root, text='Próximas actividades\nculturales gratuitas',
                     command= lambda: free_or_not_free_activities(1), bg='#c9f0b4',
                     height = 3, width = 24, font=("Arial", 13)) # button to call the 'values' command above 
canvas1.create_window(165, 330, window=button5)


button6 = tk.Button (root, text='Próximas actividades\nculturales de pago',
                     command= lambda: free_or_not_free_activities(0), bg='#c9f0b4',
                     height = 3, width = 24, font=("Arial", 13)) # button to call the 'values' command above 
canvas1.create_window(165, 210, window=button6)


button7 = tk.Button (root, text='Mostrar 500 actividades culturales\naleatorias de la base de datos',
                     command= lambda: next_global_points(7), bg='#5ee7e5',
                     height = 3, width = 28, font=("Arial", 13)) # button to call the 'values' command above 
canvas1.create_window(440, 210, window=button7)


root.mainloop()

