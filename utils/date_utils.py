"""
Módulo para manejo avanzado de fechas y horas.
Incluye soporte para expresiones relativas y reconocimiento de fechas.
"""
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any
import pytz
import re
from dateparser import parse as parse_date

# Configuración de zona horaria por defecto
DEFAULT_TIMEZONE = 'America/Argentina/Buenos_Aires'

def get_current_datetime(timezone: str = None) -> datetime:
    """Obtiene la fecha y hora actual en la zona horaria especificada."""
    tz = pytz.timezone(timezone) if timezone else pytz.timezone(DEFAULT_TIMEZONE)
    return datetime.now(tz)

def parse_datetime(text: str, reference_date: datetime = None) -> Optional[datetime]:
    """
    Parsea una cadena de texto a un objeto datetime.
    
    Args:
        text: Texto a parsear (ej: "mañana a las 3pm", "en 2 horas")
        reference_date: Fecha de referencia para cálculos relativos (default: ahora)
        
    Returns:
        datetime: Objeto datetime con la fecha/hora parseada
        None: Si no se pudo parsear la fecha
    """
    if not text:
        return None
        
    # Normalizar el texto
    text = text.lower().strip()
    
    # Manejar expresiones comunes
    if text == 'ahora':
        return get_current_datetime()
        
    # Configuración para el parser de fechas
    settings = {
        'languages': ['es'],
        'TIMEZONE': DEFAULT_TIMEZONE,
        'RETURN_AS_TIMEZONE_AWARE': True
    }
    
    if reference_date:
        settings['RELATIVE_BASE'] = reference_date
    
    # Intentar parsear la fecha
    try:
        parsed_date = parse_date(text, settings=settings)
        if parsed_date:
            # Asegurarse de que la fecha tenga zona horaria
            if parsed_date.tzinfo is None:
                tz = pytz.timezone(DEFAULT_TIMEZONE)
                parsed_date = tz.localize(parsed_date)
            return parsed_date
    except Exception as e:
        print(f"Error al parsear fecha: {e}")
    
    return None

def format_datetime(dt: datetime, format_str: str = None) -> str:
    """
    Formatea un objeto datetime a string.
    
    Args:
        dt: Fecha a formatear
        format_str: Formato (por defecto: "%d/%m/%Y %H:%M")
        
    Returns:
        str: Fecha formateada
    """
    if not dt:
        return ""
        
    if format_str is None:
        format_str = "%d/%m/%Y %H:%M"
        
    # Asegurarse de que la fecha tenga zona horaria
    if dt.tzinfo is None:
        tz = pytz.timezone(DEFAULT_TIMEZONE)
        dt = tz.localize(dt)
    
    return dt.strftime(format_str)

def is_future_datetime(dt: datetime) -> bool:
    """Verifica si una fecha es futura."""
    if not dt:
        return False
    now = get_current_datetime()
    return dt > now

def get_time_until(dt: datetime) -> str:
    """
    Obtiene una cadena legible con el tiempo restante hasta la fecha especificada.
    Ej: "en 2 horas y 30 minutos"
    """
    if not dt:
        return ""
        
    now = get_current_datetime()
    delta = dt - now
    
    if delta.total_seconds() < 0:
        return "ya pasó"
        
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    
    parts = []
    if days > 0:
        parts.append(f"{days} día{'s' if days > 1 else ''}")
    if hours > 0:
        parts.append(f"{hours} hora{'s' if hours > 1 else ''}")
    if minutes > 0 and days == 0:  # Solo mostramos minutos si no hay días
        parts.append(f"{minutes} minuto{'s' if minutes > 1 else ''}")
    
    if not parts:
        return "en menos de un minuto"
        
    return "en " + " y ".join(parts)
