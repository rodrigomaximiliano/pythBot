"""
Constantes para las respuestas del chat.
"""
from typing import Dict, List, Any

# Respuestas generales
GREETINGS = [
    "¡Hola! ¿En qué puedo ayudarte hoy?",
    "¡Hola! ¿Cómo estás?",
    "¡Hola! ¿En qué puedo asistirte?",
]

GOODBYES = [
    "¡Hasta luego! Que tengas un excelente día.",
    "¡Hasta pronto! Si necesitas algo más, aquí estaré.",
    "¡Adiós! Fue un placer ayudarte.",
]

HELP = """
Puedo ayudarte con las siguientes tareas:

* Crear recordatorios: "Recuérdame llamar a Juan mañana a las 10am"
* Agendar eventos: "Agenda una reunión el viernes a las 3pm"
* Responder preguntas: "¿Qué tiempo hará mañana?"
* Y mucho más. ¿En qué necesitas ayuda?
"""

# Mapeo de intenciones a respuestas
INTENT_RESPONSES: Dict[str, Any] = {
    "greeting": {
        "responses": GREETINGS,
        "suggestions": ["Crear recordatorio", "Ver mis eventos", "Ayuda"]
    },
    "goodbye": {
        "responses": GOODBYES,
        "suggestions": []
    },
    "help": {
        "responses": [HELP],
        "suggestions": ["Crear recordatorio", "Ver mis eventos"]
    },
    "unknown": {
        "responses": [
            "No estoy seguro de entenderte. ¿Podrías reformularlo?",
            "No he podido entender tu solicitud. ¿En qué más puedo ayudarte?",
            "Vaya, no estoy seguro de cómo responder a eso. ¿Puedes decírmelo de otra forma?"
        ],
        "suggestions": ["Necesito ayuda", "Ver ejemplos"]
    },
    "reminder_created": {
        "responses": ["He creado un recordatorio para {date}: {text}"],
        "suggestions": ["Ver mis recordatorios", "Crear otro recordatorio"]
    },
    "event_created": {
        "responses": ["He creado el evento '{title}' para el {date}"],
        "suggestions": ["Ver mis eventos", "Crear otro evento"]
    },
    "no_reminders": {
        "responses": ["No tienes recordatorios programados."],
        "suggestions": ["Crear recordatorio", "Ver ayuda"]
    },
    "no_events": {
        "responses": ["No tienes eventos programados."],
        "suggestions": ["Crear evento", "Ver ayuda"]
    }
}

# Palabras clave para detectar intenciones
INTENT_KEYWORDS = {
    "greeting": ["hola", "buenos días", "buenas tardes", "buenas noches", "saludos"],
    "goodbye": ["adiós", "chao", "hasta luego", "hasta pronto", "nos vemos"],
    "help": ["ayuda", "qué puedes hacer", "cómo funciona", "qué comandos hay"],
    "create_reminder": ["recuérdame", "recordar", "recordatorio", "avísame"],
    "list_reminders": ["mis recordatorios", "qué recordatorios tengo", "ver recordatorios"],
    "create_event": ["evento", "cita", "reunión", "agenda", "agendar", "planificar"],
    "list_events": ["mis eventos", "qué eventos tengo", "ver agenda"]
}

def get_random_response(intent: str, **kwargs) -> Dict[str, Any]:
    """Obtiene una respuesta aleatoria para una intención."""
    import random
    
    if intent not in INTENT_RESPONSES:
        intent = "unknown"
    
    response_data = INTENT_RESPONSES[intent]
    response = random.choice(response_data["responses"])
    
    # Formatear la respuesta con los parámetros proporcionados
    if kwargs:
        response = response.format(**kwargs)
    
    return {
        "response": response,
        "suggestions": response_data.get("suggestions", []),
        "intent": intent
    }
