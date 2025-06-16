"""
Módulo para manejar los comandos específicos del bot.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from .date_utils import get_current_datetime, format_datetime, parse_datetime, get_time_until
from .recordatorios import agregar_recordatorio, obtener_recordatorios, Recordatorio
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
    text = reminder_data.get('text', '').strip()
    time_expr = reminder_data.get('time_expression', '').strip()
    
    if not text and not time_expr:
        return "No entendí qué quieres que recuerde. ¿Podrías ser más específico?"
    
    # Si no hay expresión de tiempo, pedir confirmación
    if not time_expr:
        return f"¿Quieres que te recuerde '{text}'? Por favor, indícame cuándo."
    
    # Intentar parsear la fecha/hora
    try:
        reminder_time = parse_datetime(time_expr)
        if not reminder_time:
            return f"No pude entender la hora/fecha '{time_expr}'. ¿Podrías ser más específico?"
            
        # Agregar el recordatorio
        success, message = agregar_recordatorio(
            session_id=session_id,
            texto=text,
            fecha=reminder_time
        )
        
        if success:
            time_until = get_time_until(reminder_time)
            return f"✅ Recordatorio configurado: {text} {time_until}."
        return f"No pude configurar el recordatorio: {message}"
        
    except Exception as e:
        return f"Ocurrió un error al configurar el recordatorio: {str(e)}"

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
        return "No tienes recordatorios pendientes. ¡Estás al día! 🎉"
    
    # Ordenar por fecha más cercana
    recordatorios_ordenados = sorted(
        recordatorios, 
        key=lambda x: x.fecha
    )[:limit]
    
    # Formatear la respuesta
    if len(recordatorios) == 1:
        respuesta = ["📌 Tienes 1 recordatorio pendiente:"]
    else:
        respuesta = [f"📌 Tienes {len(recordatorios_ordenados)} recordatorios pendientes:"]
    
    for i, recordatorio in enumerate(recordatorios_ordenados, 1):
        tiempo_restante = get_time_until(recordatorio.fecha)
        respuesta.append(
            f"{i}. {recordatorio.texto} ({tiempo_restante} - {format_datetime(recordatorio.fecha)})"
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

# Mapeo de intenciones a manejadores
HANDLERS = {
    IntentType.GREETING: lambda *_: handle_greeting(),
    IntentType.FAREWELL: lambda *_: handle_farewell(),
    IntentType.TIME_QUERY: lambda *_: handle_time_query(),
    IntentType.DATE_QUERY: lambda *_: handle_date_query(),
    IntentType.CREATE_REMINDER: lambda data, session_id: handle_create_reminder(data, session_id),
    IntentType.LIST_REMINDERS: lambda _, session_id: handle_list_reminders(session_id),
    IntentType.HELP: lambda *_: handle_help(),
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
