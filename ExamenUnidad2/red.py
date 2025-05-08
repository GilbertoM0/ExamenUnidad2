import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

#Hago mi dataset
mxn = np.random.uniform(1000, 50000, 1000)
usd = mxn / 17.8

#hago lo de 80% y el 20%
# 80% para entrenamiento y 20% para validación
mxn_train, mxn_val, usd_train, usd_val = train_test_split(mxn, usd, test_size=0.2)


modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=[1], activation='relu'),
    tf.keras.layers.Dense(1)
])

modelo.compile(optimizer='adam', loss='mse')

# Entrenare 
history = modelo.fit(mxn_train, usd_train, epochs=30, validation_data=(mxn_val, usd_val), verbose=0)

# Predicciones
train_pred = modelo.predict(mxn_train)
val_pred = modelo.predict(mxn_val)

# Calcular accuracy 
def calcular_accuracy(y_real, y_pred, tolerancia=0.1):
    correctas = np.abs(y_real - y_pred.flatten()) <= tolerancia
    return np.mean(correctas) * 100  

train_acc = calcular_accuracy(usd_train, train_pred)
val_acc = calcular_accuracy(usd_val, val_pred)

print(f"\nAccuracy entrenamiento: {train_acc:.2f}%")
print(f"Accuracy validación:    {val_acc:.2f}%")

# Datos nuevos que no han sido vistos durante el entrenamiento
nuevos_datos = np.array([15000, 25000]) 

# Realizar las predicciones
predicciones = modelo.predict(nuevos_datos)

# Mostrar las predicciones
for mxn, usd_pred in zip(nuevos_datos, predicciones):
    print(f"${mxn:.2f} MXN → ${usd_pred[0]:.2f} USD (esperado: {mxn / 17.8:.2f})")

# Guardar el modelo
modelo.save("modelo_conversion.keras")
