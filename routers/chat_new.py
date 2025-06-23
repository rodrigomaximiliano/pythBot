import parsedatetime as pdt
import uuid
from fastapi import APIRouter, HTTPException, Form, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Tuple, Any, Literal, Union
import logging
import traceback
from datetime import datetime, timedelta
import re
from enum import Enum
import json
import os
import sys
import numpy as np
import random

# Añadir el directorio raíz al path para importaciones
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importaciones de transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Importar el motor de NLP
from nlp.intent_classifier import process_message, nlp_engine

# Tipos de mensajes
class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# Estados de la conversación
class ConversationState:
    """Clase para manejar el estado de la conversación con memoria mejorada."""
    CURRENT_TASK: Optional[str] = None
    PENDING_INFO: Dict[str, Any] = {}
    LAST_INTENT: Optional[str] = None
    CONTEXT: Dict[str, Any] = {}
    CONVERSATION_HISTORY: List[Dict[str, Any]] = []
    USER_PREFERENCES: Dict[str, Any] = {}
    MAX_HISTORY: int = 20  # Máximo de mensajes a recordar
    
    @classmethod
    def add_to_history(cls, role: str, message: str, intent: str = None):
        """Añade un mensaje al historial de la conversación."""
        entry = {
            'role': role,  # 'user' o 'assistant'
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'intent': intent
        }
        cls.CONVERSATION_HISTORY.append(entry)
        
        # Mantener solo los últimos MAX_HISTORY mensajes
        if len(cls.CONVERSATION_HISTORY) > cls.MAX_HISTORY:
            cls.CONVERSATION_HISTORY = cls.CONVERSATION_HISTORY[-cls.MAX_HISTORY:]
    
    @classmethod
    def get_conversation_context(cls, max_tokens: int = 500) -> str:
        """Obtiene el contexto de la conversación como texto."""
        context_parts = []
        current_tokens = 0
        
        # Incluir preferencias del usuario
        if cls.USER_PREFERENCES:
            prefs = ", ".join(f"{k}: {v}" for k, v in cls.USER_PREFERENCES.items())
            context_parts.append(f"Preferencias del usuario: {prefs}")
        
        # Incluir información pendiente si existe
        if cls.PENDING_INFO:
            pending = ", ".join(f"{k}: {v}" for k, v in cls.PENDING_INFO.items() 
                               if k not in ['session_id', 'last_prompt', 'prompt_timestamp'])
            if pending:
                context_parts.append(f"Información pendiente: {pending}")
        
        # Incluir el historial de la conversación
        for entry in reversed(cls.CONVERSATION_HISTORY):
            entry_text = f"{entry['role'].capitalize()}: {entry['message']}"
            if entry.get('intent'):
                entry_text += f" (Intención: {entry['intent']})"
                
            if current_tokens + len(entry_text.split()) > max_tokens:
                break
                
            context_parts.append(entry_text)
            current_tokens += len(entry_text.split())
        
        return "\n".join(reversed(context_parts))
    
    @classmethod
    def reset(cls):
        """Reinicia el estado de la conversación, pero mantiene las preferencias."""
        cls.CURRENT_TASK = None
        cls.PENDING_INFO = {}
        cls.LAST_INTENT = None
        cls.CONTEXT = {}
        cls.CONVERSATION_HISTORY = []

# Intenciones del usuario
class Intent(str, Enum):
    GREETING = "greeting"
    FAREWELL = "farewell"
    THANKS = "thanks"
    HELP = "help"
    CREATE_REMINDER = "create_reminder"
    LIST_REMINDERS = "list_reminders"
    DELETE_REMINDER = "delete_reminder"
    DATE_QUERY = "date_query"
    UNKNOWN = "unknown"

# Entidades que podemos extraer
class EntityType(str, Enum):
    TIME = "time"
    DATE = "date"
    EVENT = "event"
    TASK = "task"
    DURATION = "duration"

# Configuración de logging
logger = logging.getLogger(__name__)

# Almacenamiento en memoria para recordatorios (en producción usar una base de datos)
reminders: Dict[str, List[Dict[str, Any]]] = {}


class IntentClassifier:
    """Clase para clasificar la intención del usuario usando un modelo de lenguaje avanzado."""
    
    def __init__(self):
        # Usamos un modelo más avanzado para mejor comprensión
        self.model_name = "dccuchile/bert-base-spanish-wwm-cased"  # Versión cased para mejor manejo de mayúsculas
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=9,  # Aumentamos el número de intenciones
            ignore_mismatched_sizes=True
        )
        
        # Mapeo de índices a intenciones con más categorías
        self.id2label = {
            0: "greeting",
            1: "farewell",
            2: "thanks",
            3: "help",
            4: "create_reminder",
            5: "list_reminders",
            6: "date_query",
            7: "affirmative",
            8: "negative"
        }
        
        # Ejemplos de entrenamiento más completos y variados
        self.examples = {
            "greeting": [
                "Hola", "Buenos días", "Buenas tardes", "Buenas noches", "Saludos",
                "Hola, ¿cómo estás?", "¿Cómo estás?", "¿Qué tal?", "¿Cómo te va?", "¿Qué cuentas?",
                "Hola, ¿qué tal estás?", "¿Cómo andas?", "¿Qué hay de nuevo?", "¿Cómo has estado?",
                "Hola de nuevo", "¡Hola! Tanto tiempo", "Buen día", "Buenas"
            ],
            "farewell": [
                "Adiós", "Hasta luego", "Nos vemos", "Chao", "Hasta pronto", "Hasta la próxima",
                "Me despido", "Hasta mañana", "Hasta la vista", "Que tengas buen día", "Hasta la próxima vez",
                "Me retiro", "Hasta otro día", "Chao, nos vemos", "Hasta la próxima, cuídate"
            ],
            "thanks": [
                "Gracias", "Muchas gracias", "Te lo agradezco", "Mil gracias", "Te agradezco",
                "Gracias por tu ayuda", "Te lo agradezco mucho", "Muy amable", "Gracias por todo",
                "Te agradezco la ayuda", "Gracias de antemano", "Mil gracias por tu ayuda"
            ],
            "help": [
                "Ayuda", "¿Qué puedes hacer?", "¿Para qué sirves?", "¿Cómo te llamas?", "¿Qué sabes hacer?",
                "¿Cómo funcionas?", "¿Qué comandos tienes?", "¿Me puedes ayudar?", "¿Qué haces?",
                "¿Cuáles son tus funciones?", "¿Qué cosas puedes hacer?", "¿Cómo te puedo usar?",
                "Explícame qué haces", "¿Qué tipo de ayuda ofreces?", "¿Cómo trabajas?"
            ],
            "create_reminder": [
                "Recuérdame comprar leche", "Agenda una reunión mañana", "Necesito recordar llamar al médico",
                "Ponme un recordatorio para el dentista", "Quiero que me recuerdes algo", "Puedes agendar una cita",
                "Necesito que me recuerdes hacer la tarea", "Agéndame una llamada con Juan",
                "Recuérdame que tengo que ir al banco", "Quiero que me avises mañana temprano",
                "No olvides que tengo que pagar el recibo", "Guarda en mi agenda el cumpleaños de Ana el 15 de julio",
                "Recuérdame que mañana a las 3pm tengo reunión", "Necesito un recordatorio para el viernes a las 5pm",
                "Añade a mi calendario la cita con el médico el próximo lunes", "Recuérdame tomar mi medicina a las 8am",
                "Quiero que me recuerdes que debo llamar a mamá esta noche", "Programa una alarma para mañana a las 7am",
                "No dejes que se me olvide la reunión del miércoles", "Dentro de una hora recuérdame que debo salir"
            ],
            "list_reminders": [
                "¿Qué recordatorios tengo?", "Muestra mis recordatorios", "Lista de recordatorios",
                "¿Qué tengo agendado?", "¿Qué tengo pendiente?", "Muéstrame mis recordatorios",
                "¿Qué recordatorios hay?", "¿Tengo algo programado?", "¿Qué eventos tengo próximos?",
                "Dime qué tengo en mi agenda", "¿Qué hay en mi calendario?", "¿Qué actividades tengo pendientes?",
                "Muéstrame mis compromisos", "¿Qué recordatorios activos tengo?", "¿Qué tengo programado para hoy?",
                "¿Qué hay para mañana en mi agenda?", "¿Qué tengo que hacer esta semana?",
                "¿Tengo algún recordatorio pendiente?", "¿Qué actividades tengo agendadas?",
                "¿Cuáles son mis próximos recordatorios?"
            ],
            "date_query": [
                "¿Qué día es mañana?", "¿Qué día es hoy?", "¿Qué día cae el próximo lunes?",
                "¿Qué día es en 3 días?", "¿Qué día será el 25 de diciembre?", "¿Qué día de la semana es hoy?",
                "¿Qué día es la semana que viene?", "¿Qué día es el próximo mes?", "¿En qué día de la semana cae Navidad?",
                "¿Qué día será dentro de un mes?", "¿Qué día fue ayer?", "¿Qué día es el primero de enero?",
                "¿Qué día de la semana es el 15 de septiembre?", "¿Qué día será el último viernes de este mes?",
                "¿En qué día cae el día de la madre este año?", "¿Qué día es el solsticio de verano?",
                "¿Qué día es el equinoccio de primavera?", "¿Qué día es el día del padre?",
                "¿Qué día es el día de la independencia?", "¿Qué día es el día de acción de gracias?"
            ],
            "affirmative": [
                "sí", "si", "claro", "por supuesto", "desde luego", "así es", "correcto", "exacto",
                "afirmativo", "de acuerdo", "ok", "okey", "vale", "perfecto", "genial", "está bien",
                "me parece bien", "por qué no", "seguro", "sin problema", "adelante", "claro que sí",
                "por supuesto que sí", "así será", "confirmo", "afirmativo", "correcto", "exactamente"
            ],
            "negative": [
                "no", "para nada", "en absoluto", "nunca", "jamás", "de ninguna manera", "ni pensarlo",
                "olvídalo", "cancelar", "no gracias", "mejor no", "en otra ocasión", "ahora no", "luego",
                "quizás después", "no quiero", "no me interesa", "no, gracias", "no por ahora", "no por el momento",
                "déjalo así", "olvídalo por ahora", "no es necesario", "no hace falta", "mejor otro día"
            ]
        }
    
    def get_embedding(self, text):
        """Obtiene la representación embebida del texto."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model.bert(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze()
    
    def predict_intent(self, text):
        """Predice la intención del texto de entrada."""
        # Calcular similitud con ejemplos de entrenamiento
        input_embedding = self.get_embedding(text)
        
        similarities = {}
        for intent, examples in self.examples.items():
            # Calcular embedding promedio de los ejemplos
            example_embeddings = [self.get_embedding(ex) for ex in examples]
            avg_embedding = torch.stack(example_embeddings).mean(dim=0)
            
            # Calcular similitud del coseno
            cos = torch.nn.CosineSimilarity(dim=0)
            similarity = cos(input_embedding, avg_embedding).item()
            similarities[intent] = similarity
        
        # Obtener la intención con mayor similitud
        predicted_intent = max(similarities.items(), key=lambda x: x[1])
        
        # Umbral de confianza
        if predicted_intent[1] < 0.5:  # Ajusta este umbral según sea necesario
            return "unknown"
            
        return predicted_intent[0]

# Inicializar el clasificador de intenciones
intent_classifier = IntentClassifier()

# Configurar el router del chat sin prefijo
router = APIRouter(tags=["chat"])

# Almacenamiento en memoria para el historial de chat y estados
chat_histories: Dict[str, List[Message]] = {}
conversation_states: Dict[str, ConversationState] = {}

# Patrones para extraer entidades
PATTERNS = {
    EntityType.TIME: [
        (r'(?:a las|a la|al|las|los)\s+(\d{1,2})(?::(\d{2}))?\s*(?:hs|h|horas|:)?\s*(am|pm)?', 'time'),
        (r'mediodía', 'time'),
        (r'medianoche', 'time'),
    ],
    EntityType.DATE: [
        (r'(hoy)', 'today'),
        (r'(mañana)', 'tomorrow'),
        (r'(pasado mañana)', 'day_after_tomorrow'),
        (r'(lunes|martes|miércoles|jueves|viernes|sábado|domingo)', 'weekday'),
        (r'(próximo|próxima|siguiente)\s+(lunes|martes|miércoles|jueves|viernes|sábado|domingo)', 'next_weekday'),
        (r'(próxima semana|siguiente semana)', 'next_week'),
        (r'el (\d{1,2})(?:/|-)(\d{1,2})(?:/|-)?(\d{2,4})?', 'date'),
    ],
    EntityType.DURATION: [
        (r'(\d+)\s*(minutos?|min|horas?|h|días?|d|semanas?|s|mes(?:es)?|años?)', 'duration'),
    ]
}

def detect_intent(message: str) -> Tuple[Intent, Dict[str, Any]]:
    """
    Detecta la intención del usuario usando el motor de NLP y extrae entidades.
    
    Args:
        message: El mensaje del usuario.
        
    Returns:
        Tuple[Intent, Dict[str, Any]]: La intención detectada y un diccionario con entidades.
    """
    try:
        logger.info(f"[DETECT_INTENT] Procesando mensaje: {message}")
        lower_message = message.lower()
        
        # Si el usuario está respondiendo a la pregunta de crear un recordatorio
        if ConversationState.PENDING_INFO.get('waiting_for_task'):
            logger.info("[DETECT_INTENT] Usuario está en modo 'waiting_for_task'")
            
            # Crear un diccionario para las entidades
            entities = {}
            
            # Extraer la tarea del mensaje
            task = message.strip()
            logger.info(f"[DETECT_INTENT] Tarea extraída: {task}")
            entities['task'] = task
            
            # Manejar referencias temporales como 'mañana', 'pasado mañana', etc.
            cal = pdt.Calendar()
            time_struct, parse_status = cal.parse(message)
            now = datetime.now()
            
            # Primero verificar si se menciona 'mañana' específicamente
            if 'mañana' in message.lower() and 'pasado' not in message.lower():
                tomorrow = now + timedelta(days=1)
                entities['date'] = tomorrow.strftime('%Y-%m-%d')
                
                # Extraer hora si se menciona
                time_match = re.search(r'(\d{1,2}):?(\d{2})?\s*(a\.?m\.?|p\.?m\.?|am|pm)?', message, re.IGNORECASE)
                if time_match:
                    hour = int(time_match.group(1))
                    minute = int(time_match.group(2) or '0')
                    period = (time_match.group(3) or '').lower()
                    
                    # Convertir a formato 24h si es necesario
                    if 'p' in period and hour < 12:
                        hour += 12
                    elif 'a' in period and hour == 12:
                        hour = 0
                        
                    entities['time'] = f"{hour:02d}:{minute:02d}"
            # Si no se mencionó 'mañana', usar el parser normal
            elif parse_status > 0:  # Si se pudo extraer una fecha/hora
                parsed_date = datetime(*time_struct[:6])
                entities['date'] = parsed_date.strftime('%Y-%m-%d')
            
            # Si no se detectó fecha, ver si hay palabras clave como 'mañana', 'pasado mañana', etc.
            if 'date' not in entities:
                date_keywords = {
                    'mañana': 1,
                    'pasado mañana': 2,
                    'hoy': 0,
                    'lunes': None,  # Se manejará con parsedatetime
                    'martes': None,
                    'miércoles': None,
                    'jueves': None,
                    'viernes': None,
                    'sábado': None,
                    'domingo': None
                }
                
                for keyword, days in date_keywords.items():
                    if keyword in lower_message:
                        if days is not None:  # Para 'hoy', 'mañana', 'pasado mañana'
                            date_obj = datetime.now() + timedelta(days=days)
                            entities['date'] = date_obj.strftime('%Y-%m-%d')
                            logger.info(f"[DETECT_INTENT] Fecha basada en palabra clave '{keyword}': {entities['date']}")
                        break  # Solo considerar la primera coincidencia
            
            # Si llegamos aquí, es porque el usuario está respondiendo a la pregunta de crear un recordatorio
            # y ya hemos extraído lo que pudimos del mensaje
            logger.info(f"[DETECT_INTENT] Entidades finales: {entities}")
            return Intent.CREATE_REMINDER, entities
        
        logger.info("[DETECT_INTENT] Procesando mensaje con el motor NLP")
        
        # Primero verificar si es un recordatorio sin usar el motor de NLP
        reminder_keywords = ['recuérdame', 'recordar', 'recordarme', 'recuerda', 'necesito recordar',
                           'agenda', 'agéndame', 'ponme', 'programa', 'programar', 'necesito',
                           'quiero recordar', 'deseo recordar', 'me gustaría recordar',
                           'podrías recordarme', 'puedes recordarme', 'tengo que recordar']
        
        is_reminder = any(keyword in lower_message for keyword in reminder_keywords)
        
        if is_reminder:
            logger.info("[DETECT_INTENT] Se detectó palabra clave de recordatorio, forzando intención CREATE_REMINDER")
            intent_str = "create_reminder"
            entities = {}
        else:
            # Procesar el mensaje con el motor de NLP
            intent_str, entities, confidence = process_message(message)
            
            # Log para depuración
            logger.info(f"[DETECT_INTENT] Intención detectada: {intent_str} (confianza: {confidence:.2f})")
            logger.info(f"[DETECT_INTENT] Entidades extraídas: {entities}")
        
        # Mapear la intención a la enumeración
        try:
            intent = Intent[intent_str.upper()]
            logger.info(f"[DETECT_INTENT] Intención mapeada: {intent}")
        except (KeyError, AttributeError):
            logger.warning(f"[DETECT_INTENT] Intención desconocida: {intent_str}")
            intent = Intent.UNKNOWN
        
        # Mejorar la extracción de tareas para recordatorios
        if (intent == Intent.CREATE_REMINDER or 
            (ConversationState.LAST_INTENT == Intent.LIST_REMINDERS and 
             any(word in lower_message for word in ['sí', 'si', 'sí, crear', 'si, crear', 'claro', 'ok', 'vale', 'sí quiero', 'si quiero']))):
            
            logger.info("[DETECT_INTENT] Procesando tarea para recordatorio")
            
            # Limpiar la tarea extraída
            if 'task' in entities and entities['task']:
                task = entities['task'].strip()
                # Eliminar signos de puntuación al final
                task = re.sub(r'[.,;!?]+$', '', task)
                entities['task'] = task
                logger.info(f"[DETECT_INTENT] Tarea limpiada: {entities['task']}")
            elif not entities.get('task'):
                # Si no se detectó una tarea, usar el mensaje completo como tarea
                entities['task'] = message.strip()
                logger.info(f"[DETECT_INTENT] Usando mensaje completo como tarea: {entities['task']}")
        
        return intent, entities
    except Exception as e:
        logger.error(f"[DETECT_INTENT] Error al detectar la intención: {str(e)}", exc_info=True)
        return Intent.UNKNOWN, {}

def extract_entities(text: str) -> Dict[str, Any]:
    """
    Extrae entidades del texto usando expresiones regulares.
    
    Args:
        text: Texto del que extraer entidades
        
    Returns:
        Dict con las entidades encontradas organizadas por tipo
    """
    entities = {}
    text_lower = text.lower()
    
    # Patrones mejorados para fechas
    date_patterns = [
        # Fechas completas (dd/mm/yyyy, dd-mm-yyyy, dd.mm.yyyy, dd de mes, etc.)
        (r'\b(0?[1-9]|[12][0-9]|3[01])[/\-\. ](0?[1-9]|1[0-2])(?:[/\-\. ](\d{2,4}))?\b', 'date'),
        (r'\b(0?[1-9]|[12][0-9]|3[01])\s+(?:de\s+)?(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)(?:\s+(?:de\s+)?(\d{2,4}))?\b', 'date'),
        
        # Días de la semana (próximo lunes, este martes, etc.)
        (r'\b(el\s+)?(pr[oó]ximo|pr[oó]xima|siguiente|este|esta|el\s+pr[oó]ximo\s+)?(lunes|martes|mi[ée]rcoles|jueves|viernes|s[áa]bado|domingo)(?:\s+que\s+viene|\s+pr[oó]ximo|\s+entrante)?\b', 'relative_date'),
        
        # Fechas relativas (mañana, pasado mañana, hoy, etc.)
        (r'\b(hoy|mañana|pasado\s+mañana|ayer|anteayer|ahora|en\s+la\s+mañana|en\s+la\s+tarde|en\s+la\s+noche|más\s+tarde)\b', 'relative_date'),
        (r'\b(esta\s+noche|esta\s+tarde|esta\s+mañana|mediodía|medianoche)\b', 'relative_time'),
        
        # Patrones para "la próxima semana", "el mes que viene", etc.
        (r'\b(la\s+)?(pr[oó]xima|siguiente)\s+(semana|fin\s+de\s+semana|mes|año)\b', 'relative_period'),
        (r'\bel\s+(mes|año)\s+que\s+viene\b', 'relative_period'),
        
        # Días específicos (el 15, el día 20, etc.)
        (r'\bel\s+(día\s+)?(\d{1,2})(?:\s+de\s+(\w+))?(?:\s+a\s+las)?\b', 'day_of_month')
    ]
    
    # Patrones para horas
    time_patterns = [
        # Formato 24h (14:30, 9:15, etc.)
        (r'\b([01]?[0-9]|2[0-3])(?:[:.]([0-5][0-9]))?\s*(?:hs?|horas?|:)?\s*([hH])?\b', 'time_24h'),
        
        # Formato 12h (2:30 pm, 9 am, etc.)
        (r'\b(1[0-2]|0?[1-9])(?::([0-5][0-9]))?\s*([ap])\.?\s*m?\.?\b', 'time_12h'),
        
        # Horas con texto (a las 3, a las 3 de la tarde, etc.)
        (r'\b(?:a\s+las?\s+)?(\d{1,2})(?::([0-5][0-9]))?\s*(?:de\s+la\s+)?(mañana|tarde|noche|madrugada)?\b', 'time_text'),
        
        # Mediodía y medianoche
        (r'\b(mediod[ií]a|medianoche)\b', 'time_special')
    ]
    
    # Buscar fechas
    for pattern, label in date_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            if 'date' not in entities:
                entities['date'] = []
            entities['date'].append({
                'value': match.group(0),
                'label': label,
                'groups': match.groups()
            })
    
    # Buscar horas
    for pattern, label in time_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            if 'time' not in entities:
                entities['time'] = []
            entities['time'].append({
                'value': match.group(0),
                'label': label,
                'groups': match.groups()
            })
    
    # Extraer tareas - Patrones mejorados
    task_patterns = [
        # Patrón para "Recuérdame [tarea] [fecha/hora]"
        r'(?:recuérdame|recordar|recordarme|recuerda|necesito recordar|agenda|agéndame|ponme|programa|programar|quiero que me recuerdes)'
        r'(?:\s+(?:que|de|el|la|algo|un|una|unos|unas|para|a|en))?\s*'
        r'(.*?)'
        r'(?=\s*(?:el|para el|para la|a las?|el día|mañana|pasado mañana|hoy|\d{1,2}(?:[:.]\d{2})?|$))',
        
        # Patrón para "[Tarea] [fecha/hora]"
        r'^(?!(?:a las?|el|para|en|mañana|hoy|\d{1,2}(?:[:.]\d{2})?|lunes|martes|mi[ée]rcoles|jueves|viernes|s[áa]bado|domingo))'
        r'([^.!?\n]*?)'
        r'(?:\s+(?:a las?|el|para el|para la|el día|mañana|pasado mañana|hoy|\d{1,2}(?:[:.]\d{2})?))',
        
        # Patrón para "Necesito que me recuerdes [tarea]"
        r'(?:necesito|quiero|deseo|me gustaría|podrías|puedes)(?:\s+que\s+me\s+recuerdes?|\s+recordarme|\s+agendar|\s+programar)'
        r'(?:\s+(?:que|de|el|la|algo|un|una|unos|unas|para|a|en))?\s*'
        r'(.*?)(?:\.|\?|$)',
        
        # Patrón para frases como "Tengo que [tarea]"
        r'(?:tengo\s+que|debo|tendr[íi]a\s+que|tendr[ée]\s+que|hay\s+que)'
        r'\s+([^.!?\n]*?)'
        r'(?=\s*(?:el|para el|para la|a las?|el día|mañana|pasado mañana|hoy|\d{1,2}(?:[:.]\d{2})?|$))'
    ]
    
    # Intentar extraer la tarea usando los patrones
    for pattern in task_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            task_text = match.group(1).strip() if match.lastindex else match.group(0).strip()
            if not task_text:
                continue
                
            # Limpiar la tarea de signos de puntuación al final
            task_text = re.sub(r'[\.,;:!?]+$', '', task_text).strip()
            
            # Eliminar cualquier entidad de tiempo/fecha que ya hayamos detectado
            for entity_type in ['date', 'time']:
                if entity_type in entities:
                    for entity in entities[entity_type]:
                        if entity['value'].lower() in task_text.lower():
                            task_text = re.sub(re.escape(entity['value']), '', task_text, flags=re.IGNORECASE)
                            task_text = ' '.join(task_text.split())  # Normalizar espacios
            
            task_text = task_text.strip()
            if task_text and len(task_text) > 2:  # Asegurarse de que la tarea tenga al menos 3 caracteres
                if 'task' not in entities:
                    entities['task'] = []
                # Evitar duplicados
                if not any(t['value'].lower() == task_text.lower() for t in entities['task']):
                    entities['task'].append({
                        'value': task_text,
                        'label': 'task',
                        'groups': match.groups()
                    })
                break  # Usar solo el primer patrón que coincida
    
    return entities

def get_spanish_weekday(weekday: int) -> str:
    """Convierte el número del día de la semana a su nombre en español."""
    weekdays = [
        "lunes", "martes", "miércoles", "jueves", 
        "viernes", "sábado", "domingo"
    ]
    return weekdays[weekday % 7]

def get_spanish_month(month: int) -> str:
    """Convierte el número del mes a su nombre en español."""
    months = [
        "enero", "febrero", "marzo", "abril", "mayo", "junio",
        "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"
    ]
    return months[month - 1]

def get_current_time() -> str:
    """Obtiene la hora actual formateada en español."""
    now = datetime.now()
    hora = now.hour
    minutos = now.minute
    
    # Formatear la hora en formato 12 horas
    periodo = "a.m." if hora < 12 else "p.m."
    hora_12 = hora % 12
    if hora_12 == 0:
        hora_12 = 12
        
    return f"Son las {hora_12}:{minutos:02d} {periodo}"

def get_date_info(date_str: str) -> str:
    """
    Obtiene información sobre una fecha específica.
    
    Args:
        date_str: Cadena con la fecha a analizar (hoy, mañana, etc.)
        
    Returns:
        str: Información sobre la fecha
    """
    now = datetime.now()
    
    # Verificar si es una consulta de hora
    if any(palabra in date_str.lower() for palabra in ['qué hora es', 'que hora es', 'dime la hora']):
        return get_current_time()
    
    # Configurar localización en español
    import locale
    try:
        locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')
    except:
        locale.setlocale(locale.LC_TIME, 'Spanish_Spain.1252')
    
    # Manejo de fechas relativas
    lower_str = date_str.lower()
    
    if 'semana pasada' in lower_str:
        date = now - timedelta(weeks=1)
        dia_semana = get_spanish_weekday(date.weekday())
        mes = get_spanish_month(date.month)
        return f"La semana pasada fue del {date.day - date.weekday()} al {date.day - date.weekday() + 6} de {mes} de {date.year}"
    
    elif 'mañana' in lower_str or 'dia siguiente' in lower_str:
        date = now + timedelta(days=1)
        dia_semana = get_spanish_weekday(date.weekday())
        mes = get_spanish_month(date.month)
        return f"Mañana es {dia_semana.capitalize()} {date.day} de {mes} de {date.year}"
    
    elif 'hoy' in lower_str:
        dia_semana = get_spanish_weekday(now.weekday())
        mes = get_spanish_month(now.month)
        return f"Hoy es {dia_semana.capitalize()} {now.day} de {mes} de {now.year}"
    
    elif 'semana que viene' in lower_str or 'próxima semana' in lower_str or 'proxima semana' in lower_str:
        next_week = now + timedelta(weeks=1)
        dia_semana = get_spanish_weekday(next_week.weekday())
        mes = get_spanish_month(next_week.month)
        next_monday = now + timedelta(days=(7 - now.weekday()) % 7)
        next_sunday = next_monday + timedelta(days=6)
        return f"La semana que viene es del {next_monday.day} al {next_sunday.day} de {mes} de {next_week.year}"
    
    elif 'pasado mañana' in lower_str:
        date = now + timedelta(days=2)
        dia_semana = get_spanish_weekday(date.weekday())
        mes = get_spanish_month(date.month)
        return f"Pasado mañana es {dia_semana.capitalize()} {date.day} de {mes} de {date.year}"
    
    elif 'ayer' in lower_str:
        date = now - timedelta(days=1)
        dia_semana = get_spanish_weekday(date.weekday())
        mes = get_spanish_month(date.month)
        return f"Ayer fue {dia_semana.capitalize()} {date.day} de {mes} de {date.year}"
    
    else:
        # Intentar parsear la fecha
        try:
            cal = pdt.Calendar()
            date, parse_status = cal.parseDT(date_str)
            if parse_status > 0:  # Si se pudo parsear la fecha
                dia_semana = get_spanish_weekday(date.weekday())
                mes = get_spanish_month(date.month)
                return f"El {dia_semana.capitalize()} {date.day} de {mes} de {date.year}"
        except:
            pass
        
        # Si no se pudo determinar la fecha, devolver la fecha actual
        dia_semana = get_spanish_weekday(now.weekday())
        mes = get_spanish_month(now.month)
        return f"Hoy es {dia_semana.capitalize()} {now.day} de {mes} de {now.year}"

def generate_response(intent: Intent, entities: Dict[str, Any], session_id: str, original_message: str = "") -> str:
    """
    Genera una respuesta basada en la intención y entidades detectadas.
    
    Args:
        intent: Intención detectada
        entities: Entidades extraídas del mensaje
        session_id: ID de la sesión
        original_message: Mensaje original del usuario (opcional)
        
    Returns:
        str: Respuesta generada
    """
    # Registrar información de depuración
    logger.info(f"[GENERATE_RESPONSE] Iniciando generación de respuesta")
    logger.info(f"[GENERATE_RESPONSE] Intención: {intent}")
    logger.info(f"[GENERATE_RESPONSE] Entidades: {entities}")
    logger.info(f"[GENERATE_RESPONSE] Mensaje original: {original_message}")
    logger.info(f"[GENERATE_RESPONSE] Última intención: {ConversationState.LAST_INTENT}")
    logger.info(f"[GENERATE_RESPONSE] PENDING_INFO: {ConversationState.PENDING_INFO}")
    
    # Convertir el mensaje a minúsculas para facilitar la comparación
    lower_message = original_message.lower() if original_message else ""
    
    # Actualizar el historial de la conversación
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    
    # Manejar la intención detectada
    lower_message = original_message.lower()
    
    # Manejar respuestas de confirmación para recordatorios
    if (ConversationState.LAST_INTENT == Intent.LIST_REMINDERS and 
        any(word in lower_message for word in ['sí', 'si', 'sí, por favor', 'si, por favor', 'claro', 'ok', 'vale'])):
        ConversationState.LAST_INTENT = Intent.CREATE_REMINDER
        # Si no se pudo extraer información, pedir más detalles
        prompt = "¿Qué te gustaría que te recuerde?"
        ConversationState.PENDING_INFO = {
            'waiting_for_task': True,
            'session_id': session_id,
            'last_prompt': prompt,
            'prompt_timestamp': datetime.now().isoformat()
        }
        return prompt
    
    # Manejar consultas sobre el clima
    if any(word in lower_message for word in [
        'hace frío', 'hace calor', 'qué temperatura', 'que temperatura',
        'cómo está el clima', 'como esta el clima', 'qué tiempo hace',
        'que tiempo hace', 'va a llover', 'hace sol', 'está nublado',
        'esta nublado', 'pronóstico', 'pronostico'
    ]):
        return "Actualmente no tengo acceso a información meteorológica en tiempo real. " \
               "¿Te gustaría que te ayude con algo más?"
    
    # Manejar respuestas cuando se espera una tarea para recordatorio
    if (ConversationState.PENDING_INFO.get('waiting_for_task') and 
        ConversationState.PENDING_INFO.get('last_prompt', '').startswith('¿Qué te gustaría que te recuerde')):
        # Si estamos esperando una tarea, forzar la intención a CREATE_REMINDER
        logger.info("[GENERATE_RESPONSE] Detectada respuesta a pregunta de recordatorio, forzando intención CREATE_REMINDER")
        intent = Intent.CREATE_REMINDER
        
        # Guardar el mensaje original para usarlo después
        reminder_text = original_message
        
        # Crear el recordatorio con la información del mensaje
        reminder = {
            'id': str(uuid.uuid4()),
            'task': reminder_text,
            'date': None,
            'time': None,
            'created_at': datetime.now().isoformat()
        }
        
        # Intentar extraer fecha y hora del mensaje
        reminder_info = parse_reminder(reminder_text)
        if reminder_info:
            if reminder_info.get('date'):
                reminder['date'] = reminder_info['date']
            if reminder_info.get('time'):
                reminder['time'] = reminder_info['time']
        
        # Ya no estamos esperando una tarea
        ConversationState.PENDING_INFO.pop('waiting_for_task', None)
        
        # Preguntar por confirmación
        ConversationState.PENDING_INFO.update({
            'pending_task': reminder,
            'waiting_confirmation': True,
            'session_id': session_id,
            'last_prompt': None  # Limpiar el último prompt
        })
        
        # Construir mensaje de confirmación
        confirmation_msg = f"¿Quieres que te recuerde: \"{reminder['task']}\""
        
        if reminder.get('date'):
            try:
                date_obj = datetime.strptime(reminder['date'], '%Y-%m-%d')
                formatted_date = date_obj.strftime('%A %d de %B').capitalize()
                confirmation_msg += f" el {formatted_date}"
            except:
                confirmation_msg += f" el {reminder['date']}"
            
            if reminder.get('time'):
                try:
                    time_obj = datetime.strptime(reminder['time'], '%H:%M')
                    formatted_time = time_obj.strftime('%I:%M %p').lstrip('0')
                    confirmation_msg += f" a las {formatted_time}"
                except:
                    confirmation_msg += f" a las {reminder['time']}"
        elif reminder.get('time'):
            try:
                time_obj = datetime.strptime(reminder['time'], '%H:%M')
                formatted_time = time_obj.strftime('%I:%M %p').lstrip('0')
                confirmation_msg += f" a las {formatted_time}"
            except:
                confirmation_msg += f" a las {reminder['time']}"
        
        confirmation_msg += "?"
        return confirmation_msg
    
    # Manejar consultas de fecha y hora
    if (intent == Intent.DATE_QUERY or 
        (not ConversationState.PENDING_INFO.get('waiting_for_task') and 
         any(word in lower_message for word in [
            'qué día', 'que dia', 
            'qué fecha', 'que fecha', 
            'mañana', 'hoy', 
            'semana que viene',
            'qué hora es', 'que hora es',
            'dime la hora'
        ]))):
        return get_date_info(original_message)
    elif intent == Intent.GREETING:
        now = datetime.now()
        hour = now.hour
        if 5 <= hour < 12:
            return "¡Buenos días! ¿En qué puedo ayudarte hoy?"
        elif 12 <= hour < 19:
            return "¡Buenas tardes! ¿En qué puedo ayudarte hoy?"
        else:
            return "¡Buenas noches! ¿En qué puedo ayudarte hoy?"
    
    elif intent == Intent.FAREWELL:
        responses = [
            "¡Hasta luego! Si necesitas algo más, aquí estaré.",
            "¡Adiós! Fue un placer ayudarte.",
            "¡Hasta pronto! No dudes en volver si necesitas ayuda.",
            "¡Nos vemos! Que tengas un excelente día."
        ]
        return random.choice(responses)
    
    elif intent == Intent.THANKS:
        responses = [
            "¡De nada! Estoy para ayudarte.",
            "¡Es un placer! ¿Neitas algo más?",
            "¡Gracias a ti! ¿En qué más puedo ayudarte?",
            "¡Para eso estoy! ¿Algo más en lo que pueda asistirte?"
        ]
        return random.choice(responses)
    
    elif intent == Intent.HELP:
        return """🤖 *¿En qué puedo ayudarte?*

📅 *Fechas y Horas*
• "¿Qué día es hoy/mañana?"
• "¿Qué hora es?"
• "¿Qué día cae el próximo lunes?"

⏰ *Recordatorios*
• "Recuérdame [tarea] [fecha/hora]"
• "¿Qué recordatorios tengo?"
• "Elimina mi recordatorio de..."

💬 *Pregúntame*
• "¿Qué puedes hacer?"
• "¿Cómo estás?"
• "Gracias"

¡Pregunta lo que necesites! 😊"""
    
    elif intent == Intent.CREATE_REMINDER:
        logger.info("[CREATE_REMINDER] Procesando intención de crear recordatorio")
        logger.info(f"[CREATE_REMINDER] Estado actual de PENDING_INFO: {ConversationState.PENDING_INFO}")
        logger.info(f"[CREATE_REMINDER] Entidades detectadas: {entities}")
        logger.info(f"[CREATE_REMINDER] Mensaje original: {original_message}")
        
        # Verificar si el usuario está respondiendo a la pregunta de qué recordar
        if ConversationState.PENDING_INFO.get('waiting_for_task'):
            logger.info("[CREATE_REMINDER] Usuario está proporcionando la tarea para el recordatorio")
            
            # Usar parse_reminder para extraer información estructurada
            reminder_info = parse_reminder(original_message)
            
            if not reminder_info or not reminder_info.get('task'):
                return "No pude entender qué quieres que te recuerde. Por favor, inténtalo de nuevo."
            
            # Verificar si el usuario canceló
            lower_message = original_message.lower()
            if any(word in lower_message for word in ['cancelar', 'olvídalo', 'no importa', 'déjalo', 'nada']):
                ConversationState.PENDING_INFO = {}
                return "De acuerdo, he cancelado la creación del recordatorio."
            
            # Crear el recordatorio con la información extraída
            reminder = {
                'id': str(uuid.uuid4()),
                'task': reminder_info['task'],
                'date': reminder_info.get('date'),
                'time': reminder_info.get('time'),
                'created_at': datetime.now().isoformat()
            }
            
            # Preguntar por confirmación
            ConversationState.PENDING_INFO = {
                'pending_task': reminder,
                'waiting_confirmation': True,
                'session_id': session_id
            }
            
            # Construir mensaje de confirmación
            confirmation_msg = f"¿Quieres que te recuerde: \"{reminder['task']}\""
            if reminder.get('date'):
                try:
                    date_obj = datetime.strptime(reminder['date'], '%Y-%m-%d')
                    formatted_date = date_obj.strftime('%A %d de %B').capitalize()
                    confirmation_msg += f" el {formatted_date}"
                except:
                    confirmation_msg += f" el {reminder['date']}"
                
                if reminder.get('time'):
                    try:
                        time_obj = datetime.strptime(reminder['time'], '%H:%M')
                        formatted_time = time_obj.strftime('%I:%M %p').lstrip('0')
                        confirmation_msg += f" a las {formatted_time}"
                    except:
                        confirmation_msg += f" a las {reminder['time']}"
            elif reminder.get('time'):
                try:
                    time_obj = datetime.strptime(reminder['time'], '%H:%M')
                    formatted_time = time_obj.strftime('%I:%M %p').lstrip('0')
                    confirmation_msg += f" a las {formatted_time}"
                except:
                    confirmation_msg += f" a las {reminder['time']}"
            
            confirmation_msg += "?"
            return confirmation_msg
        
        # Si ya hay una tarea pendiente de confirmación
        elif ConversationState.PENDING_INFO.get('waiting_confirmation'):
            logger.info("[CREATE_REMINDER] Hay una tarea pendiente de confirmación")
            lower_message = original_message.lower()
            
            # Si el usuario confirma con 'sí' o similar
            if any(word in lower_message for word in ['sí', 'si', 'sí, guarda', 'si, guarda', 'claro', 'ok', 'vale', 'sí, guardar', 'si, guardar', 'sí, guardalo', 'si, guardalo', 'sí, quiero', 'si quiero']):
                try:
                    # Guardar el recordatorio
                    reminder = ConversationState.PENDING_INFO['pending_task']
                    
                    # Inicializar recordatorios para la sesión si no existen
                    session_id = ConversationState.PENDING_INFO.get('session_id', 'default')
                    if session_id not in reminders:
                        reminders[session_id] = []
                    
                    # Agregar el recordatorio
                    reminder_id = str(uuid.uuid4())
                    reminder['id'] = reminder_id
                    reminder['created_at'] = datetime.now().isoformat()
                    reminder['notified'] = False
                    
                    reminders[session_id].append(reminder)
                    
                    # Limpiar el estado pendiente
                    ConversationState.PENDING_INFO = {}
                    
                    # Construir mensaje de confirmación
                    confirmation_msg = "✅ ¡Recordatorio guardado! "
                    if reminder.get('date') or reminder.get('time'):
                        confirmation_msg += "Te recordaré "
                        if reminder.get('date'):
                            try:
                                date_obj = datetime.strptime(reminder['date'], '%Y-%m-%d')
                                formatted_date = date_obj.strftime('%A %d de %B').capitalize()
                                confirmation_msg += f"el {formatted_date}"
                            except:
                                confirmation_msg += f"el {reminder['date']}"
                            
                            if reminder.get('time'):
                                confirmation_msg += " a las "
                        
                        if reminder.get('time'):
                            try:
                                time_obj = datetime.strptime(reminder['time'], '%H:%M')
                                formatted_time = time_obj.strftime('%I:%M %p').lstrip('0')
                                confirmation_msg += formatted_time
                            except:
                                confirmation_msg += reminder['time']
                        
                        confirmation_msg += "."
                    else:
                        confirmation_msg += "No se especificó fecha ni hora."
                    
                    return confirmation_msg
                except Exception as e:
                    logger.error(f"Error al guardar el recordatorio: {str(e)}")
                    return "❌ Hubo un error al guardar el recordatorio. Por favor, inténtalo de nuevo."
            
            # Si el usuario cancela
            elif any(word in lower_message for word in ['no', 'cancelar', 'olvídalo', 'mejor no', 'no, gracias']):
                ConversationState.PENDING_INFO = {}
                return "❌ De acuerdo, no guardaré el recordatorio. ¿En qué más puedo ayudarte?"
            
            # Si no se entiende la respuesta
            else:
                # Volver a pedir confirmación
                reminder = ConversationState.PENDING_INFO.get('pending_task', {})
                task = reminder.get('task', 'esto')
                return f"No estoy seguro de tu respuesta. ¿Quieres que guarde el recordatorio para " + \
                       f"'{task}'? Responde 'sí' para confirmar o 'no' para cancelar."
        
        # Si solo falta la hora, preguntar por ella
        if date_info and not time_info:
            return f"¿A qué hora te gustaría que te recuerde \"{task}\" el {date_info}? (por ejemplo: 'a las 3 de la tarde')"
            
        # Si solo falta la fecha, preguntar por ella
        if time_info and not date_info:
            ConversationState.PENDING_INFO = {'pending_task': {'task': task, 'time': time_info}, 'session_id': session_id}
            return f"¿Para qué día quieres que te recuerde \"{task}\" a las {time_info}? (por ejemplo: 'mañana' o 'el lunes')"
            
        # Si el usuario no proporcionó una tarea clara
        if not task or task.lower() in ['algo', 'algo?', 'algo.']:
            return "No entendí qué quieres que te recuerde. Por favor, dime algo como: 'Recuérdame comprar leche mañana a las 5pm'"
        
        # Si tenemos toda la información, pedir confirmación
        ConversationState.PENDING_INFO = {'pending_task': reminder, 'waiting_confirmation': True, 'session_id': session_id}
        
        # Construir mensaje de confirmación claro
        confirm_message = "📋 *Confirmación de recordatorio*\n\n"
        confirm_message += f"📌 *Tarea*: {task}\n"
        
        # Formatear fecha
        if date_info:
            try:
                date_obj = datetime.strptime(reminder.get('date'), '%Y-%m-%d')
                formatted_date = date_obj.strftime('%A %d de %B').capitalize()
                confirm_message += f"📅 *Fecha*: {formatted_date}\n"
            except:
                confirm_message += f"📅 *Fecha*: {date_info}\n"
        
        # Formatear hora
        if time_info:
            try:
                time_obj = datetime.strptime(reminder.get('time'), '%H:%M')
                formatted_time = time_obj.strftime('%I:%M %p').lstrip('0').lower()
                confirm_message += f"⏰ *Hora*: {formatted_time}\n"
            except:
                confirm_message += f"⏰ *Hora*: {time_info}\n"
        
        confirm_message += "\n¿Es correcto? Responde 'sí' para confirmar o 'no' para cancelar."
        
        return confirm_message
    
    elif intent == Intent.LIST_REMINDERS:
        logger.info("[LIST_REMINDERS] Verificando si el usuario está respondiendo a la pregunta de crear recordatorio")
        logger.info(f"[LIST_REMINDERS] Última intención: {ConversationState.LAST_INTENT}")
        logger.info(f"[LIST_REMINDERS] Mensaje del usuario: {lower_message}")
        
        # Verificar si el usuario está respondiendo a la pregunta de crear un recordatorio
        if ConversationState.LAST_INTENT == Intent.LIST_REMINDERS and any(word in lower_message for word in ['sí', 'si', 'sí, crear', 'si, crear', 'claro', 'ok', 'vale', 'sí quiero', 'si quiero']):
            logger.info("[LIST_REMINDERS] Usuario quiere crear un recordatorio")
            ConversationState.LAST_INTENT = Intent.CREATE_REMINDER
            # Guardar que el usuario quiere crear un recordatorio
            ConversationState.PENDING_INFO = {'waiting_for_task': True}
            logger.info(f"[LIST_REMINDERS] Nuevo estado PENDING_INFO: {ConversationState.PENDING_INFO}")
            return "¡Perfecto! ¿Qué te gustaría que te recuerde? Por ejemplo: 'Recuérdame comprar leche mañana'"
        
        # Verificar si hay recordatorios
        if 'reminders' not in globals() or not reminders:
            logger.info("[LIST_REMINDERS] No hay recordatorios, preguntando si quiere crear uno")
            # Si no hay recordatorios, preguntar si quiere crear uno
            ConversationState.LAST_INTENT = Intent.LIST_REMINDERS
            return "No tienes recordatorios programados. ¿Te gustaría crear uno? (responde 'sí' o 'no')"
            
        user_reminders = [r for r in reminders.values() if r.get('session_id') == session_id]
        if not user_reminders:
            ConversationState.LAST_INTENT = Intent.LIST_REMINDERS
            return "No tienes recordatorios programados. ¿Te gustaría crear uno? (responde 'sí' o 'no')"
        
        reminders_list = []
        for i, r in enumerate(user_reminders, 1):
            when = []
            if r.get('date'):
                when.append(r['date'])
            if r.get('time'):
                when.append(r['time'])
            when_str = f" ({', '.join(when)})" if when else ""
            status = "✅ " if r.get('completed', False) else ""
            reminders_list.append(f"{i}. {status}{r['task']}{when_str}")
        
        return "📋 **Tus recordatorios:**\n" + "\n".join(reminders_list)
    
    # Si no se reconoce la intención, usar el historial para contexto
    last_messages = [msg['content'] for msg in chat_histories[session_id][-3:]]
    
    # Verificar si es una pregunta sobre la hora/fecha
    if any(word in ' '.join(last_messages).lower() for word in ['hora', 'fecha', 'día', 'dias', 'mañana', 'ayer']):
        now = datetime.now()
        return f"Hoy es {now.strftime('%A %d de %B de %Y')} y son las {now.strftime('%H:%M')}."
    
    # Respuesta por defecto
    default_responses = [
        "No estoy seguro de entenderte. ¿Podrías reformular tu pregunta?",
        "No tengo una respuesta para eso. ¿Puedes decírmelo de otra forma?",
        "Voy a necesitar más información para ayudarte con eso.",
        "¿Podrías ser más específico? No estoy seguro de entenderte."
    ]
    return random.choice(default_responses)

class ChatMessage(BaseModel):
    message: str
    session_id: str = "default"

def parse_reminder(message: str) -> Optional[Dict[str, Any]]:
    """
    Intenta extraer información de recordatorio del mensaje.
    
    Ejemplos de entradas soportadas:
    - "Recuérdame comprar leche mañana a las 5pm"
    - "Agenda una reunión el lunes a las 10am"
    - "Necesito recordar llamar al médico el viernes"
    - "Pon recordatorio de pagar el alquiler el día 5"
    - "Añade a mi agenda la cita con Juan el 15 de julio a las 3pm"
    - "No olvides que tengo que hacer la tarea para mañana"
    - "Recuérdame que mañana a las 8am tengo reunión"
    """
    from datetime import datetime, timedelta
    import re
    
    # Hacer una copia del mensaje original para no modificarlo
    original_message = message.strip()
    message_lower = original_message.lower()
    
    # Inicializar el recordatorio
    reminder = {
        'task': original_message,  # Por defecto, todo el mensaje es la tarea
        'date': None,
        'time': None,
        'original_message': original_message
    }
    
    # Lista de prefijos comunes para recordatorios con sus variantes
    reminder_patterns = [
        r'(?:recu[eé]rdame|recuerda|recordar|recordarme|recordadme|recordad|recordar[mt]e|recordar[mt]elo|recordar[mt]ela|recordar[mt]elos|recordar[mt]elas|recordar[mt]en|recordar[mt]eme|recordar[mt]enos|recordar[mt]ele|recordar[mt]eles|recordar[mt]ela|recordar[mt]elas|recordar[mt]enlo|recordar[mt]enla|recordar[mt]enlos|recordar[mt]enlas|recordar[mt]enme|recordar[mt]ennos|recordar[mt]enle|recordar[mt]enles|recordar[mt]enla|recordar[mt]enlas|recordar[mt]enlo|recordar[mt]enlos|recordar[mt]enla|recordar[mt]enlas)',
        r'(?:agenda|agéndame|agendar|programa|programar|programame|programar[mt]e|programar[mt]elo|programar[mt]ela|programar[mt]elos|programar[mt]elas|programar[mt]en|programar[mt]eme|programar[mt]enos|programar[mt]ele|programar[mt]eles|programar[mt]ela|programar[mt]elas|programar[mt]enlo|programar[mt]enla|programar[mt]enlos|programar[mt]enlas|programar[mt]enme|programar[mt]ennos|programar[mt]enle|programar[mt]enles|programar[mt]enla|programar[mt]enlas|programar[mt]enlo|programar[mt]enlos|programar[mt]enla|programar[mt]enlas)',
        r'(?:pon|ponme|poner|poner[mt]e|poner[mt]elo|poner[mt]ela|poner[mt]elos|poner[mt]elas|poner[mt]en|poner[mt]eme|poner[mt]enos|poner[mt]ele|poner[mt]eles|poner[mt]ela|poner[mt]elas|poner[mt]enlo|poner[mt]enla|poner[mt]enlos|poner[mt]enlas|poner[mt]enme|poner[mt]ennos|poner[mt]enle|poner[mt]enles|poner[mt]enla|poner[mt]enlas|poner[mt]enlo|poner[mt]enlos|poner[mt]enla|poner[mt]enlas)',
        r'(?:necesito recordar|tengo que recordar|quiero que me recuerdes|me gustaría recordar|quiero recordar|deseo recordar|debo recordar|tendría que recordar|necesitaría recordar|necesitaré recordar|necesitar[aeá]s recordar|necesitar[aeá] recordar|necesitar[aeá]is recordar|necesitar[íi]amos recordar|necesitar[íi]an recordar|necesitar[íi]a recordar|necesitar[íi]as recordar|necesitar[íi]ais recordar|necesitar[íi]amos recordar|necesitar[íi]an recordar|necesitar[íi]a recordar|necesitar[íi]as recordar|necesitar[íi]ais recordar|necesitar[íi]amos recordar|necesitar[íi]an recordar)',
        r'(?:no olvides|no olvidar|no te olvides|no olvides de|no olviden|no olvidéis|no olviden|no olvidéis de|no olvidar de|no olvidéis de|no olvidar[mt]e|no olvidar[mt]elo|no olvidar[mt]ela|no olvidar[mt]elos|no olvidar[mt]elas|no olvidar[mt]en|no olvidar[mt]eme|no olvidar[mt]enos|no olvidar[mt]ele|no olvidar[mt]eles|no olvidar[mt]ela|no olvidar[mt]elas|no olvidar[mt]enlo|no olvidar[mt]enla|no olvidar[mt]enlos|no olvidar[mt]enlas|no olvidar[mt]enme|no olvidar[mt]ennos|no olvidar[mt]enle|no olvidar[mt]enles|no olvidar[mt]enla|no olvidar[mt]enlas|no olvidar[mt]enlo|no olvidar[mt]enlos|no olvidar[mt]enla|no olvidar[mt]enlas)',
        r'(?:añade a mi agenda|añadir a mi agenda|agrega a mi agenda|agregar a mi agenda|pon en mi agenda|poner en mi agenda|mete en mi agenda|meter en mi agenda|incluye en mi agenda|incluir en mi agenda|añadir[mt]e a mi agenda|añadir[mt]elo a mi agenda|añadir[mt]ela a mi agenda|añadir[mt]elos a mi agenda|añadir[mt]elas a mi agenda|añadir[mt]en a mi agenda|añadir[mt]eme a mi agenda|añadir[mt]enos a mi agenda|añadir[mt]ele a mi agenda|añadir[mt]eles a mi agenda|añadir[mt]ela a mi agenda|añadir[mt]elas a mi agenda|añadir[mt]enlo a mi agenda|añadir[mt]enla a mi agenda|añadir[mt]enlos a mi agenda|añadir[mt]enlas a mi agenda|añadir[mt]enme a mi agenda|añadir[mt]ennos a mi agenda|añadir[mt]enle a mi agenda|añadir[mt]enles a mi agenda|añadir[mt]enla a mi agenda|añadir[mt]enlas a mi agenda|añadir[mt]enlo a mi agenda|añadir[mt]enlos a mi agenda|añadir[mt]enla a mi agenda|añadir[mt]enlas a mi agenda)'
    ]
    
    # Buscar y eliminar el prefijo de recordatorio
    clean_message = original_message
    for pattern in reminder_patterns:
        match = re.search(pattern, message_lower, re.IGNORECASE)
        if match:
            clean_message = original_message[match.end():].strip(" ,.:;\t\n")
            break
    
    # Si no se encontró un prefijo, usar el mensaje completo
    if clean_message == original_message:
        logger.info("[PARSE_REMINDER] No se encontró prefijo de recordatorio, usando mensaje completo")
    else:
        logger.info(f"[PARSE_REMINDER] Mensaje después de limpiar prefijo: '{clean_message}'")
    
    # Usar parsedatetime para extraer fechas y horas
    cal = pdt.Calendar()
    time_struct, parse_status = cal.parse(clean_message)
    
    if parse_status:
        parsed_date = datetime(*time_struct[:6])
        logger.info(f"[PARSE_REMINDER] Fecha/hora detectada: {parsed_date}")
        
        # Extraer fecha si se detecta alguna referencia temporal
        date_keywords = ['hoy', 'mañana', 'pasado mañana', 'lunes', 'martes', 'miércoles', 
                        'jueves', 'viernes', 'sábado', 'domingo', 'día', 'semana', 'mes',
                        'próximo', 'siguiente', 'que viene', 'próxima']
        
        # Verificar si hay palabras clave de fecha en el mensaje
        has_date_keyword = any(keyword in clean_message.lower() for keyword in date_keywords)
        
        # Extraer fecha si hay palabra clave de fecha o si se detectó una fecha específica
        if has_date_keyword or time_struct[0] != 0 or time_struct[1] != 0 or time_struct[2] != 0:
            reminder['date'] = parsed_date.strftime('%Y-%m-%d')
            logger.info(f"[PARSE_REMINDER] Fecha extraída: {reminder['date']}")
        
        # Extraer hora si está presente
        if time_struct[3] != 0 or time_struct[4] != 0 or time_struct[5] != 0:
            reminder['time'] = parsed_date.strftime('%H:%M')
            logger.info(f"[PARSE_REMINDER] Hora extraída: {reminder['time']}")
    else:
        logger.info("[PARSE_REMINDER] No se pudo extraer fecha/hora con parsedatetime")
    
    # Si no se detectó fecha/hora, intentar con expresiones regulares
    if not reminder['date'] and not reminder['time']:
        logger.info("[PARSE_REMINDER] Intentando extraer fecha/hora con expresiones regulares")
        
        # Patrones para fechas
        date_patterns = [
            (r'(?:el\s+)?(\d{1,2})(?:/|-|\s+de\s+)(\d{1,2})(?:/|-|\s+de\s+)?(\d{2,4})?', '%d/%m/%Y'),  # DD/MM/YYYY o DD/MM/YY
            (r'(?:el\s+)?(\d{1,2})\s+de\s+(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)(?:\s+de\s+(\d{2,4}))?', '%d de %B %Y'),  # DD de MES de YYYY
            (r'(?:el\s+)?(lunes|martes|mi[eé]rcoles|jueves|viernes|s[aá]bado|domingo)(?:\s+que\s+viene)?', '%A')  # Lunes, Martes, etc.
        ]
        
        # Patrones para horas
        time_patterns = [
            (r'a\s+las?\s+(\d{1,2})(?::(\d{2}))?\s*(?:h(?:oras?)?\s*)?(?:y\s+(\d{1,2})\s*min(?:utos?)?)?\s*(am|pm|a\.m\.|p\.m\.)?', '%I:%M %p'),  # a las 3pm, a las 15:30, etc.
            (r'(\d{1,2})(?::(\d{2}))?\s*(?:h(?:oras?)?\s*)?(?:y\s+(\d{1,2})\s*min(?:utos?)?)?\s*(am|pm|a\.m\.|p\.m\.)?', '%I:%M %p'),  # 3pm, 15:30, etc.
            (r'mediod[ií]a', '12:00'),  # mediodía
            (r'medianoche', '00:00')  # medianoche
        ]
        
        # Buscar fechas
        for pattern, date_format in date_patterns:
            match = re.search(pattern, clean_message, re.IGNORECASE)
            if match:
                try:
                    date_str = match.group(0)
                    # Intentar parsear la fecha
                    parsed_date = datetime.strptime(date_str, date_format)
                    reminder['date'] = parsed_date.strftime('%Y-%m-%d')
                    logger.info(f"[PARSE_REMINDER] Fecha extraída con regex: {reminder['date']}")
                    break
                except (ValueError, AttributeError) as e:
                    logger.warning(f"[PARSE_REMINDER] Error al parsear fecha '{date_str}': {e}")
        
        # Buscar horas
        for pattern, time_format in time_patterns:
            match = re.search(pattern, clean_message, re.IGNORECASE)
            if match:
                try:
                    time_str = match.group(0)
                    # Si es mediodía o medianoche, ya tenemos el formato correcto
                    if time_str.lower() in ['mediodía', 'medianoche']:
                        reminder['time'] = time_format
                    else:
                        # Intentar parsear la hora
                        parsed_time = datetime.strptime(time_str, time_format)
                        reminder['time'] = parsed_time.strftime('%H:%M')
                    logger.info(f"[PARSE_REMINDER] Hora extraída con regex: {reminder['time']}")
                    break
                except (ValueError, AttributeError) as e:
                    logger.warning(f"[PARSE_REMINDER] Error al parsear hora '{time_str}': {e}")
    
    # Mejorar la extracción de la tarea
    if clean_message and clean_message != original_message:
        # Si limpiamos prefijos, usar el mensaje limpio como tarea
        reminder['task'] = clean_message
    
    # Eliminar partes de fecha/hora de la tarea si es necesario
    if reminder['date'] or reminder['time']:
        # Lista de palabras relacionadas con fechas y horas
        date_phrases = [
            'hoy', 'mañana', 'pasado mañana', 'ayer', 'anteayer',
            'lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo',
            'próximo', 'próxima', 'siguiente', 'que viene', 'entrante',
            'enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
            'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre',
            'día', 'semana', 'mes', 'año', 'años', 'días', 'semanas', 'meses',
            'próximos', 'próximas', 'siguientes', 'semanas', 'meses', 'años', 'días'
        ]
        
        time_phrases = [
            'a las', 'a la', 'al', 'las', 'los', 'a.m.', 'am', 'p.m.', 'pm',
            'de la mañana', 'de la tarde', 'de la noche', 'de la madrugada',
            'mediodía', 'medianoche', 'hora', 'horas', 'minuto', 'minutos',
            'en punto', 'en punto de la mañana', 'en punto de la tarde', 'en punto de la noche',
            'de la mañana', 'de la tarde', 'de la noche', 'de la madrugada'
        ]
        
        # Eliminar partes de fecha/hora de la tarea
        task_parts = []
        words = reminder['task'].split()
        i = 0
        n = len(words)
        
        while i < n:
            word = words[i].lower()
            
            # Verificar si la palabra es parte de una frase de tiempo
            time_phrase_found = False
            for phrase in time_phrases + date_phrases:
                if ' ' in phrase:  # Si es una frase de múltiples palabras
                    phrase_words = phrase.split()
                    if i + len(phrase_words) <= n and ' '.join(words[i:i+len(phrase_words)]).lower() == phrase:
                        i += len(phrase_words) - 1  # Saltar las palabras de la frase
                        time_phrase_found = True
                        break
                elif word == phrase:
                    time_phrase_found = True
                    break
            
            # Si no es una palabra de tiempo, agregarla a la tarea
            if not time_phrase_found and not any(p in word for p in ['/', '-', ':']) and not word.replace(':', '').replace('.', '').isdigit():
                task_parts.append(words[i])
            
            i += 1
        
        # Unir las partes limpias de la tarea
        cleaned_task = ' '.join(task_parts).strip(" ,.:;\t\n")
        
        # Si después de limpiar la tarea queda vacía, mantener la original
        if cleaned_task:
            reminder['task'] = cleaned_task
        
        logger.info(f"[PARSE_REMINDER] Tarea después de limpiar: '{reminder['task']}'")
    
    # Si la tarea está vacía después de procesar, devolver None
    if not reminder['task'].strip():
        logger.warning("[PARSE_REMINDER] La tarea está vacía después de procesar")
        return None
    
    logger.info(f"[PARSE_REMINDER] Recordatorio procesado: {reminder}")
    return reminder

def check_and_notify_reminders():
    """
    Verifica los recordatorios próximos y simula el envío de notificaciones.
    En una aplicación real, esto se ejecutaría periódicamente en segundo plano.
    """
    now = datetime.now()
    notification_threshold = timedelta(minutes=15) # Notificar recordatorios en los próximos 15 minutos

    logger.info("Verificando recordatorios próximos...")

    # Iterar sobre todas las sesiones y sus recordatorios
    for session_id, user_reminders in reminders.items():
        reminders_to_notify = []
        # Crear una copia de la lista para poder modificar el estado 'notified' sin problemas durante la iteración
        reminders_copy = user_reminders[:]
        for reminder in reminders_copy:
            # Asegurarse de que 'date_time' es un objeto datetime y 'notified' existe
            if isinstance(reminder.get("date_time"), datetime) and not reminder.get("notified", False):
                time_difference = reminder["date_time"] - now

                # Si el recordatorio está en el futuro y dentro del umbral de notificación
                if time_difference > timedelta(seconds=0) and time_difference <= notification_threshold:
                    # Simular envío de notificación (imprimir en log)
                    logger.info(f"SIMULANDO NOTIFICACIÓN para sesión {session_id}: Recordatorio próximo: '{reminder['task']}' en {time_difference}")
                    # Marcar como notificado (esto no es persistente con el almacenamiento en memoria)
                    # En una implementación con DB, actualizarías el registro aquí.
                    reminder["notified"] = True
                    reminders_to_notify.append(reminder)
        
        # Opcional: podrías hacer algo con reminders_to_notify si quisieras agrupar notificaciones

    logger.info("Verificación de recordatorios completada.")


@router.get("/chat/test")
async def test_endpoint():
    """Endpoint de prueba para verificar que la API está funcionando."""
    # Llamar a la función de verificación de recordatorios (para demostración)
    check_and_notify_reminders()
    return {"status": "success", "message": "Chat endpoint is working! Reminders checked."}

@router.get("/chat/reminders")
async def list_reminders(session_id: str = "default"):
    """Endpoint para listar los recordatorios de una sesión."""
    try:
        user_reminders = reminders.get(session_id, [])
        # Formatear las fechas para una mejor visualización e incluir estado de notificación
        formatted_reminders = []
        for r in user_reminders:
            formatted_reminders.append({
                "task": r["task"],
                "date_time": r["date_time"].isoformat() if isinstance(r["date_time"], datetime) else str(r["date_time"]),
                "notified": r.get("notified", False)
            })
        return {
            "status": "success",
            "reminders": formatted_reminders,
            "count": len(formatted_reminders)
        }
    except Exception as e:
        logger.error(f"Error al listar recordatorios: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/")
async def chat_endpoint(
    message: str = Form(...),
    session_id: str = Form("default")
):
    """
    Endpoint principal para el chat.
    Procesa el mensaje del usuario, detecta la intención, extrae entidades
    y genera una respuesta utilizando el contexto de la conversación.
    """
    try:
        logger.info("\n" + "="*80)
        logger.info(f"[CHAT_ENDPOINT] Nueva solicitud recibida")
        logger.info(f"[CHAT_ENDPOINT] Mensaje recibido: {message}")
        logger.info(f"[CHAT_ENDPOINT] Session ID: {session_id}")
        
        # Verificar si hay información pendiente de una conversación anterior
        if (ConversationState.PENDING_INFO and 
            'session_id' in ConversationState.PENDING_INFO and 
            ConversationState.PENDING_INFO['session_id'] != session_id):
            logger.info("[CHAT_ENDPOINT] Nueva sesión detectada, reiniciando estado...")
            ConversationState.reset()
        
        # Actualizar el ID de sesión actual
        if 'session_id' not in ConversationState.PENDING_INFO:
            ConversationState.PENDING_INFO['session_id'] = session_id
        
        # Añadir mensaje del usuario al historial
        ConversationState.add_to_history('user', message)
        
        # Obtener contexto de la conversación
        context = ConversationState.get_conversation_context()
        logger.info(f"[CHAT_ENDPOINT] Contexto de la conversación: {context}")
        
        # Detectar la intención y extraer entidades
        try:
            logger.info("[CHAT_ENDPOINT] Detectando intención y entidades...")
            intent, entities = detect_intent(message)
            logger.info(f"[CHAT_ENDPOINT] Intención detectada: {intent}")
            logger.info(f"[CHAT_ENDPOINT] Entidades extraídas: {entities}")
        except Exception as e:
            logger.error(f"[CHAT_ENDPOINT] Error al detectar la intención: {e}")
            intent, entities = Intent.UNKNOWN, {}
        
        # Generar la respuesta
        logger.info("[CHAT_ENDPOINT] Generando respuesta...")
        response = generate_response(
            intent=intent,
            entities=entities,
            session_id=session_id,
            original_message=message
        )
        
        logger.info(f"[CHAT_ENDPOINT] Respuesta generada: {response[:100]}...")
        
        # Añadir respuesta al historial
        ConversationState.add_to_history('assistant', response, intent.value if intent else None)
        
        # Actualizar la última intención
        if intent:
            ConversationState.LAST_INTENT = intent
        
        # Verificar si hay recordatorios para notificar
        check_and_notify_reminders()
        
        return {
            "response": response,
            "intent": intent.value if intent else "unknown",
            "session_id": session_id,
            "context": context[-500:] if context else "",
            "entities": entities
        }
        
    except Exception as e:
        error_msg = f"Error en el endpoint de chat: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Error al procesar el mensaje",
                "details": str(e)
            }
        )
