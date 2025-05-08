from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__, static_folder='static', template_folder='templates')


try:
    modelo = tf.keras.models.load_model("modelo_conversion.keras", compile=False)
    modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])
except Exception as e:
    print(f"Error al cargar el modelo: {e}")

# Ruta principal
@app.route('/')
def inicio():
    return render_template('index.html')

@app.route('/convertir', methods=['POST'])
def convertir():
    try:
        data = request.get_json()
        mxn = float(data.get('mxn', 0))

        if mxn <= 0:
            return jsonify({"error": "El valor de MXN debe ser mayor a 0"}), 400

        pred = modelo.predict(np.array([[mxn]]), verbose=0)
        return jsonify({"usd": float(pred[0][0])})
    except KeyError:
        return jsonify({"error": "Debe proporcionar un valor de 'mxn'"}), 400
    except ValueError:
        return jsonify({"error": "El valor proporcionado debe ser un número"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Ejecutar servidor
if __name__ == '__main__':
    app.run(debug=True)
