"""
Módulo para manejar los comandos específicos del bot.
"""
from typing import Dict, Any, List, Optional
import random # Importar el módulo random
from datetime import datetime
from .date_utils import get_current_datetime, format_datetime, parse_datetime, get_time_until
from .recordatorios import agregar_recordatorio, obtener_recordatorios, Recordatorio
from .eventos import agregar_evento, obtener_eventos, Evento # Importar funciones de eventos
from .intent_recognizer import IntentType

def handle_greeting() -> str:
    """Maneja el saludo inicial."""
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "¡Buenos días! ¿En qué puedo ayudarte hoy?"
    elif 12 <= hour < 19:
        return "¡Buenas tardes! ¿En qué puedo ayudarte?"
    else:
        return "¡Buenas noches! ¿En qué puedo ayudarte?"

def handle_farewell() -> str:
    """Maneja la despedida."""
    return "¡Hasta luego! Que tengas un excelente día. 😊"

def handle_time_query() -> str:
    """Responde con la hora actual."""
    now = get_current_datetime()
    return f"Son las {format_datetime(now, time_format='%H:%M')}."

def handle_date_query() -> str:
    """Responde con la fecha actual."""
    now = get_current_datetime()
    return f"Hoy es {format_datetime(now, date_format='%A, %d de %B de %Y')}."

def handle_create_reminder(reminder_data: Dict[str, Any], session_id: str) -> str:
    """
    Crea un nuevo recordatorio.
    
    Args:
        reminder_data: Datos del recordatorio (texto y expresión de tiempo)
        session_id: ID de la sesión del usuario
        
    Returns:
        str: Mensaje de confirmación
    """
    text = reminder_data.get('task', '').strip() # Usar la clave 'task'
    reminder_time = reminder_data.get('date_time') # Obtener directamente el objeto datetime
    
    if not text and reminder_time is None:
        return "No entendí qué quieres que recuerde. ¿Podrías ser más específico?"
    
    # Si no hay fecha/hora, pedir confirmación
    if reminder_time is None:
        return f"¿Quieres que te recuerde '{text}'? Por favor, indícame cuándo."
    
    # Agregar el recordatorio
    try:
        success, message = agregar_recordatorio(
            session_id=session_id,
            texto=text,
            fecha=reminder_time # Usar directamente el objeto datetime
        )
        
        if success:
            time_until = get_time_until(reminder_time)
            formatted_date = format_datetime(reminder_time)
            return f"✅ ¡Listo! Te recordaré '{text}' {time_until} ({formatted_date})."
        return f"Parece que hubo un problema al configurar el recordatorio: {message}"

    except Exception as e:
        return f"¡Oh no! Ocurrió un error inesperado al configurar el recordatorio: {str(e)}"

def handle_list_reminders(session_id: str, limit: int = 5) -> str:
    """
    Lista los recordatorios pendientes.
    
    Args:
        session_id: ID de la sesión del usuario
        limit: Número máximo de recordatorios a mostrar
        
    Returns:
        str: Lista formateada de recordatorios
    """
    recordatorios = obtener_recordatorios(session_id, solo_activos=True)
    
    if not recordatorios:
        return "¡Excelente! No tienes recordatorios pendientes. ✨"

    # Ordenar por fecha más cercana
    recordatorios_ordenados = sorted(
        recordatorios, 
        key=lambda x: x.fecha
    )[:limit]
    
    # Formatear la respuesta
    if len(recordatorios_ordenados) == 1: # Usar recordatorios_ordenados para el conteo
        respuesta = ["📌 Tienes 1 recordatorio pendiente:"]
    else:
        respuesta = [f"📌 Tienes {len(recordatorios_ordenados)} recordatorios pendientes:"]

    for i, recordatorio in enumerate(recordatorios_ordenados, 1):
        tiempo_restante = get_time_until(recordatorio.fecha)
        formatted_date = format_datetime(recordatorio.fecha)
        respuesta.append(
            f"{i}. {recordatorio.texto} ({tiempo_restante} - {formatted_date})"
        )

    return "\n".join(respuesta)

def handle_help() -> str:
    """Muestra los comandos disponibles."""
    return """
🤖 *Comandos disponibles:*

• *Saludo*: "Hola", "Buenos días", "Buenas tardes"
• *Hora*: "¿Qué hora es?", "Dime la hora"
• *Fecha*: "¿Qué día es hoy?", "¿En qué fecha estamos?"
• *Recordatorios*:
  - "Recuérdame [algo] [cuándo]"
  - "Ponme un recordatorio para [algo] [cuándo]"
  - "¿Qué recordatorios tengo?"
• *Ayuda*: "¿Qué puedes hacer?", "Ayuda"

Ejemplos:
- "Recuérdame llamar a Juan mañana a las 3pm"
- "Ponme un recordatorio para la reunión el viernes a las 10am"
- "¿Qué recordatorios tengo pendientes?"
"""

def handle_create_event(event_data: Dict[str, Any], session_id: str) -> str:
    """
    Crea un nuevo evento.

    Args:
        event_data: Datos del evento (título, fecha/hora, ubicación, descripción)
        session_id: ID de la sesión del usuario

    Returns:
        str: Mensaje de confirmación
    """
    title = event_data.get('task', event_data.get('title', 'un evento')).strip() # Usar 'task' o 'title'
    time_expr = event_data.get('date_time', '').strip() # Obtener la cadena de fecha/hora
    ubicacion = event_data.get('ubicacion', '').strip()
    descripcion = event_data.get('descripcion', '').strip()

    if not title and not time_expr:
        return "No entendí qué evento quieres crear. ¿Podrías ser más específico?"

    if not time_expr:
        return f"¿Quieres crear el evento '{title}'? Por favor, indícame cuándo."

    # Intentar parsear la fecha/hora
    try:
        event_time = parse_datetime(time_expr)
        if not event_time:
            return f"No pude entender la hora/fecha '{time_expr}' para el evento. ¿Podrías ser más específico?"

        # Agregar el evento
        success, message = agregar_evento(
            session_id=session_id,
            titulo=title,
            fecha=event_time,
            ubicacion=ubicacion if ubicacion else None,
            descripcion=descripcion if descripcion else None
        )

        if success:
            time_until = get_time_until(event_time)
            formatted_date = format_datetime(event_time)
            response_msg = f"✅ ¡Evento creado! '{title}' {time_until} ({formatted_date})."
            if ubicacion:
                response_msg += f" En: {ubicacion}."
            if descripcion:
                response_msg += f" Descripción: {descripcion}."
            return response_msg
        return f"Parece que hubo un problema al configurar el evento: {message}"

    except Exception as e:
        return f"¡Oh no! Ocurrió un error inesperado al configurar el evento: {str(e)}"

def handle_list_events(session_id: str) -> str:
    """
    Lista los eventos pendientes.

    Args:
        session_id: ID de la sesión del usuario

    Returns:
        str: Lista formateada de eventos
    """
    eventos = obtener_eventos(session_id, solo_activos=True)

    if not eventos:
        return "¡Genial! No tienes eventos próximos. ✨"

    # Ordenar por fecha más cercana
    eventos_ordenados = sorted(
        eventos,
        key=lambda x: x.fecha
    )

    # Formatear la respuesta
    if len(eventos_ordenados) == 1: # Usar eventos_ordenados para el conteo
        respuesta = ["🗓️ Tienes 1 evento pendiente:"]
    else:
        respuesta = [f"🗓️ Tienes {len(eventos_ordenados)} eventos pendientes:"]

    for i, evento in enumerate(eventos_ordenados, 1):
        tiempo_restante = get_time_until(evento.fecha)
        formatted_date = format_datetime(evento.fecha)
        line = f"{i}. {evento.titulo} ({tiempo_restante} - {formatted_date})"
        if evento.ubicacion:
            line += f" en {evento.ubicacion}"
        if evento.descripcion:
            line += f" ({evento.descripcion})"
        respuesta.append(line)


    return "\n".join(respuesta)


# Mapeo de intenciones a manejadores
HANDLERS = {
    IntentType.GREETING: lambda data, session_id: handle_greeting(), # Ajustar lambda para consistencia
    IntentType.FAREWELL: lambda data, session_id: handle_farewell(), # Ajustar lambda para consistencia
    IntentType.TIME_QUERY: lambda data, session_id: handle_time_query(), # Ajustar lambda para consistencia
    IntentType.DATE_QUERY: lambda data, session_id: handle_date_query(), # Ajustar lambda para consistencia
    IntentType.CREATE_REMINDER: lambda data, session_id: handle_create_reminder(data, session_id),
    IntentType.LIST_REMINDERS: lambda data, session_id: handle_list_reminders(session_id), # Ajustar lambda para consistencia
    IntentType.CREATE_EVENT: lambda data, session_id: handle_create_event(data, session_id), # Nuevo manejador de eventos
    IntentType.LIST_EVENTS: lambda data, session_id: handle_list_events(session_id), # Nuevo manejador de eventos
    IntentType.HELP: lambda data, session_id: handle_help(), # Ajustar lambda para consistencia
    # Agregar placeholder para clima
    IntentType.WEATHER_QUERY: lambda data, session_id: "La funcionalidad del clima está en desarrollo. ¡Pronto podrás consultarlo! ☀️",
    IntentType.AFFIRMATION: lambda data, session_id: handle_affirmation() # Nuevo manejador para afirmaciones
}

def process_intent(intent_type, intent_data: Dict[str, Any], session_id: str) -> str:
    """
    Procesa una intención y devuelve la respuesta apropiada.
    
    Args:
        intent_type: Tipo de intención
        intent_data: Datos adicionales de la intención
        session_id: ID de la sesión del usuario
        
    Returns:
        str: Respuesta al usuario
    """
    handler = HANDLERS.get(intent_type)
    if not handler:
        return "Lo siento, no entendí lo que quieres decir. ¿Podrías reformularlo?"
    
    try:
        return handler(intent_data, session_id)
    except Exception as e:
        return f"¡Ups! Ocurrió un error: {str(e)}"

def handle_affirmation() -> str:
   """Maneja una afirmación simple."""
   responses = [
       "Entendido.",
       "De acuerdo.",
       "Claro.",
       "Perfecto.",
       "Sí."
   ]
   return random.choice(responses)
