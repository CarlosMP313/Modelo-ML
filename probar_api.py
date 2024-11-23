import requests
import email
from email.mime.text import MIMEText

# Dirección de la API
URL_API = "http://127.0.0.1:5000/clasificar"

def enviar_correo_y_clasificar(asunto, cuerpo, remitente="test@example.com", destinatario="receiver@example.com"):
    try:
        # Crear un correo en formato MIME
        mensaje = MIMEText(cuerpo, "plain")
        mensaje["Subject"] = asunto
        mensaje["From"] = remitente
        mensaje["To"] = destinatario

        # Enviar el correo a la API para clasificación
        response = requests.post(URL_API, data=mensaje.as_string())
        if response.status_code == 200:
            resultado = response.json()
            print(f"Resultado de la clasificación: {resultado['resultado']}")
        else:
            print(f"Error en la clasificación: {response.json()}")
    except Exception as e:
        print(f"Error al enviar el correo: {e}")

# Prueba
enviar_correo_y_clasificar(
    asunto="¡Oferta especial!",
    cuerpo="Hola, estimado cliente, ¡Felicidades! por su.Por favor, haga clic en el siguiente enlace para reclamar su recompensa Oferta válida solo por tiempo limitado.",
)
