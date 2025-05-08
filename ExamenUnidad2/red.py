import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


mxn = np.random.uniform(1000, 50000, 1000)
usd = mxn / 17.8


mxn_train, mxn_val, usd_train, usd_val = train_test_split(mxn, usd, test_size=0.2)


modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=[1], activation='relu'),
    tf.keras.layers.Dense(1)
])

modelo.compile(optimizer='adam', loss='mse')

# 4. Entrenare 
history = modelo.fit(mxn_train, usd_train, epochs=30, validation_data=(mxn_val, usd_val), verbose=0)

# 5. Predicciones
train_pred = modelo.predict(mxn_train)
val_pred = modelo.predict(mxn_val)

# 6. Calcular accuracy personalizado
def calcular_accuracy(y_real, y_pred, tolerancia=0.1):
    correctas = np.abs(y_real - y_pred.flatten()) <= tolerancia
    return np.mean(correctas) * 100  # porcentaje

train_acc = calcular_accuracy(usd_train, train_pred)
val_acc = calcular_accuracy(usd_val, val_pred)

print(f"\nAccuracy entrenamiento: {train_acc:.2f}%")
print(f"Accuracy validación:    {val_acc:.2f}%")


# 7. Guardar el modelo
modelo.save("modelo_conversion.keras")