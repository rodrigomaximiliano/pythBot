from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datetime import datetime
import speech_recognition as sr
import tempfile
from typing import Optional

from config import messages
from utils.ia_utils import generar_respuesta_ia, reforzar_respuesta_ia
from utils.chat_history import get_history, append_history
from utils.recordatorios import agregar_recordatorio, obtener_recordatorios_futuros, total_recordatorios
from utils.nlp_utils import extraer_info_recordatorio, extraer_info_evento, detectar_intencion

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatMessage(BaseModel):
    message: str = None

# Carga el modelo y el tokenizer solo una vez al iniciar
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# Historial simple de chat por sesión (en memoria)
chat_histories = {}

def obtener_texto(message: Optional[str], audio: Optional[UploadFile]) -> Optional[str]:
    """
    Devuelve el texto extraído del mensaje o del audio.
    """
    if audio is not None:
        recognizer = sr.Recognizer()
        with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmp:
            tmp.write(audio.file.read())
            tmp.flush()
            with sr.AudioFile(tmp.name) as source:
                audio_data = recognizer.record(source)
                try:
                    return recognizer.recognize_google(audio_data, language="es-ES")
                except Exception:
                    return None
    elif message:
        return message.strip().lower()
    return None

@router.post("/")
async def chat(
    message: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    session_id: str = Form("default")
):
    now = datetime.now()
    text = None

    # Robustez: ignora audio vacío o inválido
    if audio is not None and getattr(audio, "filename", None):
        if not audio.filename or audio.filename == "" or audio.file is None:
            audio = None

    if audio is not None:
        audio.file.seek(0)
        text = await obtener_texto_async(message, audio)
        if text is None:
            return {"response": messages.ERROR_AUDIO}
    else:
        text = obtener_texto(message, None)
        if text is None:
            return {"response": messages.ERROR_INPUT}

    # Comandos rápidos (ya implementados)
    if text in ["ayuda", "help"]:
        return {
            "response": (
                "Puedes pedirme que cree recordatorios o eventos, o que te muestre tu agenda. "
                "Ejemplo: 'Recuérdame llamar a Juan mañana a las 10', 'Agéndame una reunión el viernes a las 15', "
                "'mis recordatorios', 'agenda'."
            )
        }
    if text in ["ejemplos", "example", "examples"]:
        return {
            "response": (
                "Ejemplos de uso:\n"
                "- Recuérdame comprar pan mañana a las 9\n"
                "- Agéndame una cita médica el lunes a las 17\n"
                "- Mostrar mis recordatorios\n"
                "- ¿Qué eventos tengo esta semana?"
            )
        }

    # Mejor detección de saludos (palabra exacta o al inicio)
    saludos = [
        "hola", "buenas", "buenos días", "buenas tardes", "buenas noches", "saludos", "hey"
    ]
    if any(text.startswith(s) or text == s for s in saludos):
        return {"response": messages.SALUDO}

    # Detecta intención usando NLP y contexto
    intencion = detectar_intencion(text)
    if intencion == "recordatorio":
        texto_limpio, fecha = extraer_info_recordatorio(text)
        if not fecha:
            return {"response": "No pude detectar la fecha del recordatorio. Por favor indica una fecha y hora clara, por ejemplo: 'mañana a las 10'."}
        if not texto_limpio:
            return {"response": "¿Qué debo recordar para esa fecha? Por favor indica el texto del recordatorio."}
        ok = agregar_recordatorio(session_id, texto_limpio, fecha)
        if not ok:
            return {"response": "No puedes crear recordatorios en el pasado. Por favor indica una fecha y hora futura."}
        return {
            "response": f"Recordatorio creado para el {fecha.strftime('%d/%m/%Y %H:%M')}: '{texto_limpio}'"
        }
    if intencion == "evento":
        titulo, descripcion, fecha = extraer_info_evento(text)
        if not fecha:
            return {"response": "No pude detectar la fecha del evento. Por favor indica una fecha y hora clara."}
        if not titulo:
            return {"response": "¿Cuál es el título del evento?"}
        # Puedes validar descripción si es importante
        return {
            "response": f"Evento creado para el {fecha.strftime('%d/%m/%Y %H:%M')}: '{titulo}'{' - ' + descripcion if descripcion else ''}"
        }

    # Si no es recordatorio ni evento, responde siempre con IA
    history = get_history(session_id)
    ultimos_recordatorios = obtener_recordatorios_futuros(session_id)[-3:]
    contexto_recordatorios = ""
    if ultimos_recordatorios:
        contexto_recordatorios = "Tus próximos recordatorios:\n" + "\n".join(
            [f"- {t} ({f.strftime('%d/%m/%Y %H:%M')})" for t, f in ultimos_recordatorios]
        ) + "\n"

    system_prompt = (
        "Eres un asistente virtual experto en recordatorios, agenda y eventos. "
        "Ayudas a los usuarios a crear, consultar y gestionar recordatorios y eventos en su calendario. "
        "Puedes responder preguntas sobre cómo usar la app, cómo crear recordatorios, eventos, y dar ejemplos claros. "
        "Si el usuario pregunta por funcionalidades, explica que puedes recibir mensajes de texto o audio, "
        "extraer fechas, horas y descripciones, y guardar recordatorios o eventos. "
        "Si el usuario pregunta por algo fuera de la lógica de la app, responde brevemente que solo puedes ayudar con recordatorios, agenda y eventos.\n"
        f"{contexto_recordatorios}"
    )

    response, chat_history_ids = generar_respuesta_ia(
        tokenizer=tokenizer,
        model=model,
        system_prompt=system_prompt,
        history=history,
        text=text
    )

    # Refuerza la respuesta si es poco útil
    response = reforzar_respuesta_ia(response, contexto=text)

    append_history(session_id, chat_history_ids)

    return {"response": response}

# Aquí puedes agregar funciones auxiliares para guardar recordatorios/eventos en la base de datos o en memoria
# def guardar_recordatorio(texto, fecha, session_id):
#     ...

# Helper async para audio
async def obtener_texto_async(message: Optional[str], audio: Optional[UploadFile]) -> Optional[str]:
    if audio is not None:
        recognizer = sr.Recognizer()
        with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmp:
            tmp.write(await audio.read())
            tmp.flush()
            with sr.AudioFile(tmp.name) as source:
                audio_data = recognizer.record(source)
                try:
                    return recognizer.recognize_google(audio_data, language="es-ES")
                except Exception:
                    return None
    elif message:
        return message.strip().lower()
    return None
