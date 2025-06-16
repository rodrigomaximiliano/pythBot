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
    HELP = "ayuda"
    UNKNOWN = "desconocido"

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
            IntentType.HELP: [
                r'ayuda', r'qué puedes hacer', r'cómo funcionas',
                r'qué comandos hay', r'qué puedo preguntar'
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
        # Limpiar el texto
        clean_text = text.lower()
        for pattern in self.patterns[IntentType.CREATE_REMINDER]:
            clean_text = re.sub(pattern, '', clean_text, flags=re.IGNORECASE)
        clean_text = clean_text.strip()
        
        # Buscar indicadores de tiempo
        time_indicators = [
            'a las', 'a la', 'el', 'para el', 'para las', 'para la',
            'mañana', 'tarde', 'noche', 'mediodía', 'medianoche',
            'hoy', 'mañana', 'pasado mañana', 'esta semana',
            'la próxima semana', 'el próximo', 'la próxima',
            'en punto', 'y media', 'y cuarto', 'menos cuarto'
        ]
        
        # Buscar la parte de tiempo en el texto
        time_part = ""
        reminder_text = clean_text
        
        for indicator in time_indicators:
            if indicator in clean_text:
                idx = clean_text.find(indicator)
                time_part = clean_text[idx:].strip()
                reminder_text = clean_text[:idx].strip()
                break
        
        # Si no se encontró indicador de tiempo, usar todo el texto como recordatorio
        if not time_part:
            reminder_text = clean_text
        
        return IntentType.CREATE_REMINDER, {
            'text': reminder_text,
            'time_expression': time_part if time_part else None
        }

# Instancia global del reconocedor de intenciones
intent_recognizer = IntentRecognizer()
