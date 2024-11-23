from flask import Flask, request, jsonify
import joblib
import email
import email.policy
from ModeloMejorado import TransformadorEmailACounterDePalabras


# Cargar el modelo y el pipeline
modelo = joblib.load("Spam_Classifier.pkl")
pipeline = joblib.load("preprocess_pipeline.pkl")

app = Flask(__name__)

@app.route("/clasificar", methods=["POST"])
def clasificar_correo():
    try:
        # Obtener el correo enviado en formato MIME
        datos_correo = request.data.decode("utf-8")
        correo = email.message_from_string(datos_correo, policy=email.policy.default)

        # Procesar el correo utilizando el pipeline
        correo_transformado = pipeline.transform([correo])
        prediccion = modelo.predict(correo_transformado)[0]

        # Responder con la clasificación
        resultado = "Spam" if prediccion == 1 else "Ham"
        return jsonify({"resultado": resultado})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
