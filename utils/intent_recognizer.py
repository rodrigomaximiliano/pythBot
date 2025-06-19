"""
Módulo para reconocer la intención del usuario en mensajes de texto.
"""
import re
from typing import Dict, Any, Tuple
from enum import Enum

class IntentType(Enum):
    """Tipos de intenciones soportadas."""
    GREETING = "saludo"
    FAREWELL = "despedida"
    TIME_QUERY = "hora_actual"
    DATE_QUERY = "fecha_actual"
    CREATE_REMINDER = "crear_recordatorio"
    LIST_REMINDERS = "listar_recordatorios"
    CREATE_EVENT = "crear_evento" # Nueva intención
    LIST_EVENTS = "listar_eventos" # Nueva intención
    WEATHER_QUERY = "consultar_clima" # Nueva intención
    HELP = "ayuda"
    UNKNOWN = "desconocido"
    AFFIRMATION = "afirmacion" # Nueva intención para afirmaciones simples

class IntentRecognizer:
    """Reconoce la intención en mensajes de texto."""
    
    def __init__(self):
        # Patrones para reconocer intenciones
        self.patterns = {
            IntentType.GREETING: [
                r'hola', r'buen(os|as)', r'hey', r'buenos d[ií]as',
                r'buenas tardes', r'buenas noches', r'holi'
            ],
            IntentType.FAREWELL: [
                r'adi[oó]s', r'chau', r'chao', r'hasta luego', 
                r'nos vemos', r'hasta pronto', r'hasta mañana'
            ],
            IntentType.TIME_QUERY: [
                r'qu[eé] hora es', r'dime la hora', r'sabes qu[eé] hora es',
                r'podr[ií]as decirme la hora', r'hora actual', r'qué horas son'
            ],
            IntentType.DATE_QUERY: [
                r'qu[eé] d[ií]a es', r'qué d[ií]a es hoy', 
                r'fecha de hoy', r'en qu[eé] d[ií]a estamos',
                r'podr[ií]as decirme la fecha', r'qué d[ií]a es hoy'
            ],
            IntentType.CREATE_REMINDER: [
                r'recu[eé]rdame (que |a\s*)?', 
                r'pon(me)? (un )?recordatorio',
                r'av[ií]same (de |para |que )?',
                r'necesito recordar',
                r'quiero que me recuerdes',
                r'programa( me)? (un )?recordatorio'
            ],
            IntentType.LIST_REMINDERS: [
                r'qu[eé] recordatorios tengo', 
                r'mu[eé]strame mis recordatorios',
                r'cu[aá]les son mis recordatorios',
                r'tengo recordatorios',
                r'list(a|ar) (mis )?recordatorios',
                r'recordatorios pendientes'
            ],
            IntentType.CREATE_EVENT: [ # Patrones para crear evento
                r'crea(r)? (un )?evento',
                r'agenda(r)? (un )?evento',
                r'programa(r)? (un )?evento',
                r'tenemos (un )?evento',
                r'organiza(r)? (un )?evento'
            ],
            IntentType.LIST_EVENTS: [ # Patrones para listar eventos
                r'qu[eé] eventos tengo',
                r'mu[eé]strame mis eventos',
                r'cu[aá]les son mis eventos',
                r'tengo eventos',
                r'list(a|ar) (mis )?eventos',
                r'eventos pendientes'
            ],
            IntentType.WEATHER_QUERY: [ # Patrones para consultar clima
                r'qu[eé] tiempo hace',
                r'c[oó]mo est[aá] el clima',
                r'temperatura',
                r'va a llover',
                r'hace fr[ií]o',
                r'hace calor',
                r'pron[oó]stico',
                r'clima en' # Para extraer ubicación
            ],
            IntentType.HELP: [
                r'ayuda', r'qué puedes hacer', r'cómo funcionas',
                r'qué comandos hay', r'qué puedo preguntar'
            ],
            IntentType.AFFIRMATION: [ # Patrones para afirmaciones simples
                r's[ií]'
            ]
        }
    
    def recognize(self, text: str) -> Tuple[IntentType, Dict[str, Any]]:
        """
        Reconoce la intención en un mensaje de texto.
        
        Args:
            text: Texto a analizar
            
        Returns:
            Tuple[IntentType, Dict]: (tipo de intención, datos adicionales)
        """
        if not text or not isinstance(text, str):
            return IntentType.UNKNOWN, {}
            
        text = text.lower().strip()
        
        # Verificar cada patrón
        for intent_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                   if intent_type == IntentType.CREATE_REMINDER:
                       return self._process_reminder(text)
                   elif intent_type == IntentType.CREATE_EVENT:
                        return self._process_event(text) # Nuevo método para procesar eventos
                   elif intent_type == IntentType.WEATHER_QUERY:
                        return self._process_weather_query(text) # Nuevo método para procesar clima
                   return intent_type, {}

        # Si no se reconoce, ver si es una pregunta sobre hora/fecha
        if re.search(r'hora|horas', text):
            return IntentType.TIME_QUERY, {}
        elif re.search(r'd[ií]a|fecha', text):
            return IntentType.DATE_QUERY, {}

        return IntentType.UNKNOWN, {}

    def _process_reminder(self, text: str) -> Tuple[IntentType, Dict[str, Any]]:
        """
        Procesa un mensaje para crear un recordatorio.
        
        Args:
            text: Texto del recordatorio
            
        Returns:
            Tuple[IntentType, Dict]: (tipo de intención, datos del recordatorio)
        """
        # Limpiar el texto de los patrones de creación de recordatorio
        clean_text = text.lower()
        for pattern in self.patterns[IntentType.CREATE_REMINDER]:
            clean_text = re.sub(pattern, '', clean_text, flags=re.IGNORECASE).strip()

        # Intentar separar la tarea de la expresión de tiempo
        # Buscar patrones de tiempo al final del texto
        time_patterns_end = [
            r'(a las \d{1,2}(:\d{2})?( ?[ap]\.?m\.?)?)',
            r'(el \d{1,2}( de \w+)?( de \d{4})?)',
            r'(mañana|tarde|noche|hoy|pasado mañana)',
            r'(pr[oó]ximo (lunes|martes|mi[eé]rcoles|jueves|viernes|s[aá]bado|domingo))',
            r'(la pr[oó]xima semana)',
            r'(\d+ (minutos?|horas?|d[ií]as?)) despu[eé]s' # Ejemplo de duración relativa
        ]

        reminder_text = clean_text
        time_part = None

        for pattern in time_patterns_end:
            match = re.search(pattern + r'$', clean_text, re.IGNORECASE)
            if match:
                time_part = match.group(0).strip()
                reminder_text = clean_text[:match.start()].strip()
                break # Encontramos una expresión de tiempo al final, salimos

        # Si no se encontró expresión de tiempo al final, buscar en el medio
        if not time_part:
             time_patterns_middle = [
                r'(a las \d{1,2}(:\d{2})?( ?[ap]\.?m\.?)?)',
                r'(el \d{1,2}( de \w+)?( de \d{4})?)',
                r'(mañana|tarde|noche|hoy|pasado mañana)',
                r'(pr[oó]ximo (lunes|martes|mi[eé]rcoles|jueves|viernes|s[aá]bado|domingo))',
                r'(la pr[oó]xima semana)'
             ]
             for pattern in time_patterns_middle:
                 match = re.search(pattern, clean_text, re.IGNORECASE)
                 if match:
                     time_part = match.group(0).strip()
                     # Intentar separar el texto antes y después de la expresión de tiempo
                     parts = re.split(pattern, clean_text, flags=re.IGNORECASE)
                     reminder_text = (parts[0] + parts[-1]).strip() # Concatenar partes antes y después
                     break # Encontramos una expresión de tiempo en el medio, salimos


        # Si después de la separación la tarea queda vacía, usar el texto original limpio
        if not reminder_text:
             reminder_text = clean_text

        return IntentType.CREATE_REMINDER, {
            'task': reminder_text, # Cambiado de 'text' a 'task' para consistencia con entidades
            'date_time': time_part if time_part else None # Cambiado de 'time_expression' a 'date_time'
        }

    def _process_event(self, text: str) -> Tuple[IntentType, Dict[str, Any]]:
        """
        Procesa un mensaje para crear un evento.

        Args:
            text: Texto del evento

        Returns:
            Tuple[IntentType, Dict]: (tipo de intención, datos del evento)
        """
        # Lógica similar a _process_reminder para extraer título, fecha/hora, ubicación, descripción
        # Esto es una implementación básica y puede necesitar mejoras
        # Limpiar el texto de los patrones de creación de evento
        clean_text = text.lower()
        for pattern in self.patterns[IntentType.CREATE_EVENT]:
            clean_text = re.sub(pattern, '', clean_text, flags=re.IGNORECASE).strip()

        event_data = {}
        event_data['title'] = clean_text # Por defecto, todo el texto limpio es el título
        event_data['date_time'] = None
        event_data['ubicacion'] = None
        event_data['descripcion'] = None

        # Intentar extraer fecha/hora, ubicación y descripción usando regex
        # Nota: La extracción de fecha/hora con regex es limitada. Idealmente se usaría un parser.

        # Buscar fecha/hora (ejemplos de patrones)
        date_time_patterns = [
            r'(el\s+\d{1,2}(?: de \w+)?(?: de \d{4})?(?:\s+a las\s+\d{1,2}(?::\d{2})?(?:\s*[ap]\.?m\.?)?)?)', # el [fecha] a las [hora]
            r'(mañana|tarde|noche|hoy|pasado mañana)', # Días relativos
            r'(pr[oó]ximo (lunes|martes|mi[eé]rcoles|jueves|viernes|s[aá]bado|domingo))', # Próximo día de la semana
            r'(la pr[oó]xima semana)', # Próxima semana
            r'(\d{1,2}(?::\d{2})?(?:\s*[ap]\.?m\.?)?)' # Solo hora
        ]

        for pattern in date_time_patterns:
            match = re.search(pattern, clean_text, re.IGNORECASE)
            if match:
                event_data['date_time'] = match.group(0).strip()
                # Eliminar la parte de fecha/hora del texto limpio para refinar el título
                clean_text = clean_text.replace(match.group(0), '').strip()
                event_data['title'] = clean_text # Actualizar título
                break # Asumimos que solo hay una expresión de fecha/hora principal

        # Buscar ubicación (ejemplo básico: buscar "en [lugar]")
        location_match = re.search(r' en (.*)', clean_text, re.IGNORECASE)
        if location_match:
            event_data['ubicacion'] = location_match.group(1).strip()
            # Eliminar la parte de ubicación del texto limpio
            clean_text = clean_text.replace(location_match.group(0), '').strip()
            event_data['title'] = clean_text # Actualizar título


        # Buscar descripción (ejemplo básico: buscar "sobre [tema]" o "para [propósito]")
        description_match = re.search(r' (sobre|para) (.*)', clean_text, re.IGNORECASE)
        if description_match:
            event_data['descripcion'] = description_match.group(2).strip()
            # Eliminar la parte de descripción del texto limpio
            clean_text = clean_text.replace(description_match.group(0), '').strip()
            event_data['title'] = clean_text # Actualizar título


        # Si después de la extracción el título queda vacío, usar el texto original limpio
        if not event_data['title']:
             event_data['title'] = text.lower().strip() # Usar el texto original limpio como fallback


        return IntentType.CREATE_EVENT, event_data

    def _process_weather_query(self, text: str) -> Tuple[IntentType, Dict[str, Any]]:
        """
        Procesa un mensaje para consultar el clima.

        Args:
            text: Texto de la consulta

        Returns:
            Tuple[IntentType, Dict]: (tipo de intención, datos de la consulta)
        """
        # Intentar extraer la ubicación de la consulta
        location = None
        location_match = re.search(r'clima en (.*)', text, re.IGNORECASE)
        if location_match:
            location = location_match.group(1).strip()

        return IntentType.WEATHER_QUERY, {'location': location}


# Instancia global del reconocedor de intenciones
intent_recognizer = IntentRecognizer()
