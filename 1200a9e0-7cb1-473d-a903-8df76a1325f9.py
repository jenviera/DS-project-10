#!/usr/bin/env python
# coding: utf-8

# Hola Jennifer! Como te va?
# 
# Mi nombre es Facundo! Un gusto conocerte, seré tu revisor en este proyecto.
# 
# A continuación un poco sobre la modalidad de revisión que usaremos:
# 
# Cuando enccuentro un error por primera vez, simplemente lo señalaré, te dejaré encontrarlo y arreglarlo tú cuenta. Además, a lo largo del texto iré haciendo algunas observaciones sobre mejora en tu código y también haré comentarios sobre tus percepciones sobre el tema. Pero si aún no puedes realizar esta tarea, te daré una pista más precisa en la próxima iteración y también algunos ejemplos prácticos. Estaré abierto a comentarios y discusiones sobre el tema.
# 
# Encontrará mis comentarios a continuación: **no los mueva, modifique ni elimine**.
# 
# Puedes encontrar mis comentarios en cuadros verdes, amarillos o rojos como este:
# 
# <div class="alert alert-block alert-success">
# <b>Comentario del revisor.</b> <a class="tocSkip"></a>
# 
# Exito. Todo se ha hecho de forma exitosa.
# </div>
# 
# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor.</b> <a class="tocSkip"></a>
# 
# Observación. Algunas recomendaciones.
# </div>
# 
# <div class="alert alert-block alert-danger">
# 
# <b>Comentario del revisor.</b> <a class="tocSkip"></a>
# 
# Necesita arreglos. Este apartado necesita algunas correcciones. El trabajo no puede ser aceptado con comentarios rojos. 
# </div>
# 
# Puedes responder utilizando esto:
# 
# <div class="alert alert-block alert-info">
# <b>Respuesta de estudiante.</b> <a class="tocSkip"></a>
# </div>

# # Introducción:

# <div class="alert alert-block alert-success">
# <b>Review General. (Iteración 1) </b> <a class="tocSkip"></a>
# 
# 
# Jennifer, siempre me tomo este tiempo al inicio de tu proyecto para comentar mis apreciaciones generales de esta iteración de tu entrega. 
#     
# 
# Me gusta comenzar dando la bienvenida al mundo de los datos a los estudiantes, te deseo lo mejor y espero que consigas lograr tus objetivos. Personalmente me gusta brindar el siguiente consejo, "Está bien equivocarse, es normal y es lo mejor que te puede pasar. Aprendemos de los errores y eso te hará mejor programadora ya que podrás descubrir cosas a medida que avances y son estas cosas las que te darán esa experiencia para ser una gran Data Scientist"
#     
# Ahora si yendo a esta notebook. Jennifer quiero felicitarte porque has hecho un gran trabajo a lo largo de toda la notebook, desde las interpretaciones hasta las implementaciones, se ha notado un gran manejo de las herramientas y una gran comprensión de los resultados obtenidos. Felicitaciones Jennifer, tu trabajo esta en las mejores condiciones para ser aprobado.
# 
# Éxitos dentro de tu camino en el mundo de los datos Jennifer, saludos!

# En la industria petrolera, la identificación de ubicaciones óptimas para la perforación de nuevos pozos es crucial para maximizar los beneficios y minimizar los riesgos financieros. OilyGiant, una compañía líder en extracción de petróleo, ha asignado la tarea de encontrar los mejores lugares para abrir 200 nuevos pozos de petróleo. Este proyecto se centra en analizar datos geológicos de tres regiones distintas para determinar la región más prometedora en términos de rentabilidad y riesgo.
# 
# **Objetivo del Proyecto:**
# 
# El objetivo principal es desarrollar un modelo predictivo que estime el volumen de reservas de petróleo en nuevos pozos y utilizar este modelo para seleccionar las mejores ubicaciones para perforar. Los pasos específicos incluyen:
# 
# 1. Lectura y Preparación de Datos: Cargar y explorar los datos geológicos de las tres regiones proporcionadas, asegurando la calidad y consistencia de los mismos.
# 2. Entrenamiento y Evaluación del Modelo: Desarrollar y validar un modelo de regresión lineal para predecir el volumen de reservas de petróleo en cada pozo.
# 3. Selección de Pozos Óptimos: Identificar los 200 pozos con las reservas estimadas más altas en cada región.
# 4. Análisis de Beneficios y Riesgos: Calcular el beneficio potencial de los pozos seleccionados y evaluar los riesgos utilizando la técnica de bootstrapping.
# 5. Recomendación de Región: Basado en el análisis de beneficios y riesgos, recomendar la región con el mayor margen de beneficio y un riesgo de pérdida aceptable.

# ## Preparación de Datos

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy import stats as st
from scipy.stats import bootstrap


# In[2]:


data_0 = pd.read_csv('/datasets/geo_data_0.csv')
data_1 = pd.read_csv('/datasets/geo_data_1.csv')
data_2 = pd.read_csv('/datasets/geo_data_2.csv')


# In[3]:


display(data_0.head())
print(data_0.info())


# In[4]:


display(data_1.head())
print(data_1.info())


# In[5]:


display(data_2.head())
print(data_2.info())


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor.</b> <a class="tocSkip"></a>
# 
# Excelente carga de los datos manteniendolos por separados de las importaciones de librerías como así la implementación de métodos para profundizar en la comprensión de nuestros datos. Bien hecho Jennifer!

# In[6]:


data_0 = data_0.drop('id', axis= 1)
data_1 = data_1.drop('id', axis= 1)
data_2 = data_2.drop('id', axis= 1)


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor.</b> <a class="tocSkip"></a>
# 
# Y nuevamente perfecta decisión la de quitar aquella feature que no aporta valor como id!

# ### Conclusiones
# 
# - Los datos geológicos de las tres regiones se cargaron y se verificó su integridad. 
# - Los conjuntos de datos no presentaron valores nulos ni inconsistencias significativas. 
# - En el proceso de preparación de los datos para nuestro análisis, se tomó la decisión de eliminar la columna "id" de nuestro conjunto de datos porque esta columna no aporta información valiosa ni contribuye significativamente a nuestros objetivos.

# ## Entrenamiento y Evaluación del Modelo

# In[7]:


# Dividir los datos en un conjunto de entrenamiento (75%) y un conjunto de validación (25%)
def split_data (data):
    features = data.drop('product', axis=1)
    target = data['product']
    
    features_train, features_valid, target_train, target_valid = train_test_split(features, target, test_size=0.25, random_state=12345)
    return features_train, features_valid, target_train, target_valid


# In[16]:


#Entrenar y evaluar el modelo
def train_model(features_train, target_train, features_valid, target_valid):
    model = LinearRegression()
    model.fit(features_train, target_train)
    
    predictions = model.predict(features_valid) 
    
    rmse = mean_squared_error(target_valid, predictions, squared=False)
    mean_prediction = predictions.mean()
    return predictions, rmse, mean_prediction


# In[17]:


# El proceso para cada conjunto de datos
for i, data in enumerate([data_0, data_1, data_2]):
    features_train, features_valid, target_train, target_valid = split_data(data)
    predictions, rmse, mean_prediction = train_model(features_train, target_train, features_valid, target_valid)
    print(f"Region {i}: RMSE = {rmse}, Mean Prediction = {mean_prediction}")


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor.</b> <a class="tocSkip"></a>
# 
# Excelente Jennifer al crear funciones a cargo de los procedimientos, esto siempre nos permite ser más eficientes con el código! Perfecta declaración del modelo junto a la separación de datos, bien hecho!

# In[11]:


#Creación de funciones para análisis 
def process_region(data, region_number):
    features_train, features_valid, target_train, target_valid = split_data(data)
    predictions, rmse, mean_prediction = train_model(features_train, target_train, features_valid, target_valid)
    print(f"Region {region_number}: RMSE = {rmse}, Mean Prediction = {mean_prediction}")
    return predictions, target_valid

predictions_0, target_valid_0 = process_region(data_0, 0)
predictions_1, target_valid_1 = process_region(data_1, 1)
predictions_2, target_valid_2 = process_region(data_2, 2)


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor.</b> <a class="tocSkip"></a>
# 
# Una vez resolvamos lo anteriormente dicho deberíamo spoder obtener estos valores. El procedimiento de la función es correcta!

# ### Conclusiones
# - Se entrenó un modelo de regresión lineal para cada región utilizando una división del 75% para entrenamiento y 25% para validación.
# - La evaluación del modelo muestra la precisión de las predicciones para cada región. Un RMSE bajo y una predicción media cercana a la realidad indican un buen desempeño del modelo.
# 
# Región 0: El RMSE fue de 37.57 miles de barriles respecto a los valores reales y la predicción media fue de 92.59 miles de barriles.
# 
# Región 1: El RMSE fue de 0.89 miles de barriles respecto a los valores reales y la predicción media fue de 68.72 miles de barriles..
# 
# Región 2: El RMSE fue de 40.02 miles de barriles respecto a los valores reales y la predicción media fue de 94.96 miles de barriles.

# ## Selección de Pozos Óptimos

# In[24]:


budget = 100000000
num_wells = 200
income_per_unit = 4.5

# Seleccionamos los top 200 pozos
def profit(predictions, actual, top_n=200): 
    top_n_indices = predictions.argsort()[-top_n:]
    top_n_actual = actual.iloc[top_n_indices]
    profit = top_n_actual.sum() * income_per_unit * 1000 - budget
    return profit

# Calculamos la ganancia potencial para cada región
profit_0 = profit(predictions_0, target_valid_0)
profit_1 = profit(predictions_1, target_valid_1)
profit_2 = profit(predictions_2, target_valid_2)

print(f"Profit for Region 0: {profit_0}")
print(f"Profit for Region 1: {profit_1}")
print(f"Profit for Region 2: {profit_2}")


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor.</b> <a class="tocSkip"></a>
# 
# Aquí el procedimiento para el calculo de ganancias es perfecto Jennifer, has tomado los top 200 pozos en base a las predicciones pero a la vez tomaste los valores reales. Luego el cálculo de ganancia es perfecto al sumar, multiplicar y restar el presupuesto. Bien hecho!

# ### Conclusiones
# - Se eligieron los 200 pozos con los valores de predicción más altos de cada una de las 3 regiones. Se calculó la ganancia de dichos 200 pozos de cada región y según los datos la región 0 es la ideal para invertir en el desarrollo de pozos petrolíferos, con una ganancia de 33 208 260.43 USD.

# ## Análisis de Beneficios y Riesgos

# In[23]:


def bootstrap_profit(predictions, y_valid, n_samples=1000):
    profits = []
    for _ in range(n_samples):
        if isinstance(predictions, np.ndarray):
            predictions = pd.Series(predictions)
        sample_indices = np.random.choice(predictions.index, size=len(predictions), replace=True)
        sample_predictions = predictions.iloc[sample_indices]
        sample_valid = y_valid.iloc[sample_indices]
        profit = calculate_profit(sample_predictions, sample_valid)
        profits.append(profit)
    
    profits = np.array(profits)
    mean_profit = profits.mean()
    confidence_interval = np.percentile(profits, [2.5, 97.5])
    risk_of_loss = (profits < 0).mean()
    
    return mean_profit, confidence_interval, risk_of_loss

mean_profit_0, ci_0, risk_0 = bootstrap_profit(predictions_0, target_valid_0)
mean_profit_1, ci_1, risk_1 = bootstrap_profit(predictions_1, target_valid_1)
mean_profit_2, ci_2, risk_2 = bootstrap_profit(predictions_2, target_valid_2)

print(f'Región 0: Beneficio medio: {mean_profit_0}, IC 95%: {ci_0}, Riesgo de pérdidas: {risk_0}')
print(f'Región 1: Beneficio medio: {mean_profit_1}, IC 95%: {ci_1}, Riesgo de pérdidas: {risk_1}')
print(f'Región 2: Beneficio medio: {mean_profit_2}, IC 95%: {ci_2}, Riesgo de pérdidas: {risk_2}')

# Seleccionar la mejor región basada en el análisis de riesgos y ganancias
best_region_risk = max(mean_profit_0, mean_profit_1, mean_profit_2)
print(f'La mejor región basada en el análisis de riesgos y ganancias es: {best_region_risk}')


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor.</b> <a class="tocSkip"></a>
# 
# Respecto al Bootstraping un excelente acercamiento inicial como a la vez excelente modularización en la función. Si quisieramos evaluar diferentes aspectos podríamos modificar el valor size para que en cada iteración tomemos diferentes tamaños de muestras. Por otro lado una perfecta elección de los pozos como a la vez calculo de ganancias, promedio de ganancias, perdidas y los intervalores inferiores como superiores.

# ### Conclusiones
# Descripción: Se calculó el beneficio potencial de los 200 pozos seleccionados en cada región y se evaluaron los riesgos utilizando la técnica de bootstrapping.
# 
# - Región 0: Beneficio promedio de A0, intervalo de confianza del 95% de CI0, riesgo de pérdida de R0.
# - Región 1: Beneficio promedio de A1, intervalo de confianza del 95% de CI1, riesgo de pérdida de R1.
# - Región 2: Beneficio promedio de A2, intervalo de confianza del 95% de CI2, riesgo de pérdida de R2.
# 
# Impacto: El análisis de beneficios y riesgos permite una evaluación completa del potencial de ganancias y la probabilidad de pérdidas. Seleccionar una región con un alto beneficio promedio y bajo riesgo de pérdida es fundamental para una inversión exitosa.

# ## Recomendación de Región

# Dado que el objetivo de la empresa es recomendar la selección de la región con el mayor margen de beneficio, podemos basar nuestra recomendación en los beneficios promedio calculados para cada región. Por lo que, según los resultados proporcionados: la Región 0 tiene el beneficio promedio más alto, seguida por la Región 2 y luego la Región 1.
# 
# Por lo tanto, la recomendación sería seleccionar la Región 0 para invertir en la apertura de los 200 pozos nuevos de petróleo. Sin embargo, es importante tener en cuenta la incertidumbre asociada con esta elección y considerar otros factores relevantes, como los costos operativos, la disponibilidad de recursos y el riesgo empresarial, antes de tomar una decisión final.
# 
# Este proyecto demostró la importancia de la combinación de técnicas de análisis de datos, modelado predictivo y evaluación de riesgos para la toma de decisiones informadas en la industria petrolera. Utilizando datos sintéticos, fuimos capaces de identificar la región más prometedora para el desarrollo de nuevos pozos petrolíferos, maximizando los beneficios potenciales y minimizando los riesgos financieros.

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor.</b> <a class="tocSkip"></a>
# 
# Felicitaciones Jennifer, un gran trabajao que demuestra una gran capacidad tanto de comprensión de lo realizado como de análisis con las conclusiones tanto parciales como finales. Muy bien hecho, felicitaciones!

# In[ ]:




