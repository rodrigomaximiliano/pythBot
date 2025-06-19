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
    CURRENT_TASK: Optional[str] = None
    PENDING_INFO: Dict[str, Any] = {}
    LAST_INTENT: Optional[str] = None

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


class IntentClassifier:
    """Clase para clasificar la intención del usuario usando BERT."""
    
    def __init__(self):
        self.model_name = "dccuchile/bert-base-spanish-wwm-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=6,  # Número de intenciones que manejamos
            ignore_mismatched_sizes=True
        )
        
        # Mapeo de índices a intenciones
        self.id2label = {
            0: "greeting",
            1: "farewell",
            2: "thanks",
            3: "help",
            4: "create_reminder",
            5: "list_reminders"
        }
        
        # Ejemplos de entrenamiento (few-shot learning)
        self.examples = {
            "greeting": [
                "Hola", "Buenos días", "Buenas tardes", "Buenas noches",
                "Hola, ¿cómo estás?", "¿Cómo estás?", "¿Qué tal?", "¿Cómo te va?",
                "Hola, ¿qué tal estás?", "¿Cómo andas?", "¿Qué hay de nuevo?"
            ],
            "farewell": [
                "Adiós", "Hasta luego", "Nos vemos", "Chao", "Hasta pronto",
                "Hasta la próxima", "Me despido", "Hasta mañana", "Hasta la vista"
            ],
            "thanks": [
                "Gracias", "Muchas gracias", "Te lo agradezco", "Mil gracias",
                "Te agradezco", "Gracias por tu ayuda", "Te lo agradezco mucho"
            ],
            "help": [
                "Ayuda", "¿Qué puedes hacer?", "¿Para qué sirves?", "¿Cómo te llamas?",
                "¿Qué sabes hacer?", "¿Cómo funcionas?", "¿Qué comandos tienes?",
                "¿Me puedes ayudar?", "¿Qué haces?", "¿Cuáles son tus funciones?"
            ],
            "create_reminder": [
                "Recuérdame comprar leche", 
                "Agenda una reunión mañana",
                "Necesito recordar llamar al médico",
                "Ponme un recordatorio para el dentista",
                "Quiero que me recuerdes algo",
                "Puedes agendar una cita",
                "Necesito que me recuerdes hacer la tarea",
                "Agéndame una llamada con Juan",
                "Recuérdame que tengo que ir al banco",
                "Quiero que me avises mañana temprano"
            ],
            "list_reminders": [
                "¿Qué recordatorios tengo?",
                "Muestra mis recordatorios",
                "Lista de recordatorios",
                "¿Qué tengo agendado?",
                "¿Qué tengo pendiente?",
                "Muéstrame mis recordatorios",
                "¿Qué recordatorios hay?",
                "¿Tengo algo programado?"
            ],
            "date_query": [
                "¿Qué día es mañana?",
                "¿Qué día es hoy?",
                "¿Qué día cae el próximo lunes?",
                "¿Qué día es en 3 días?",
                "¿Qué día será el 25 de diciembre?",
                "¿Qué día de la semana es hoy?",
                "¿Qué día es la semana que viene?",
                "¿Qué día es el próximo mes?",
                "¿Qué día es mañana?",
                "¿Qué día es hoy?",
                "¿Qué día es mañana?",
                "¿Qué día es hoy?",
                "¿Qué día es mañana?",
                "¿Qué día es hoy?",
                "¿Qué día es mañana?",
                "¿Qué día es hoy?"
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

# Almacenamiento en memoria para el historial de chat, recordatorios y estados
chat_histories: Dict[str, List[Message]] = {}
reminders: Dict[str, List[Dict]] = {}
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
        # Procesar el mensaje con el motor de NLP
        intent_str, entities, confidence = process_message(message)
        
        # Log para depuración
        logger.info(f"Intención detectada: {intent_str} (confianza: {confidence:.2f})")
        logger.info(f"Entidades extraídas: {entities}")
        
        # Mapear la intención a la enumeración
        try:
            intent = Intent[intent_str.upper()]
        except (KeyError, AttributeError):
            intent = Intent.UNKNOWN
        
        # Mejorar la extracción de tareas para recordatorios
        if intent == Intent.CREATE_REMINDER and 'task' in entities:
            # Limpiar la tarea extraída
            task = entities['task'].strip()
            # Eliminar signos de puntuación al final
            task = re.sub(r'[.,;!?]+$', '', task)
            entities['task'] = task
        
        return intent, entities
    
    except Exception as e:
        logger.error(f"Error en detect_intent: {str(e)}\n{traceback.format_exc()}")
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
    
    # Extraer fecha y hora primero
    for entity_type, patterns in PATTERNS.items():
        for pattern, label in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if entity_type not in entities:
                    entities[entity_type.value] = []
                entities[entity_type.value].append({
                    "value": match.group(0),
                    "label": label,
                    "groups": match.groups()
                })
    
    # Extraer tareas para recordatorios
    if "puedes agendar" in text_lower or "puedes recordar" in text_lower:
        # Extraer todo lo que viene después de "puedes agendar/recordar"
        task_match = re.search(
            r'(?:puedes\s+(?:agendar|recordar))(?:\s+(?:que|de|el|la|algo|un|una|unos|unas))?\s*(.*?)(?:\.|\?|$)', 
            text_lower
        )
        if task_match and task_match.group(1).strip():
            task_text = task_match.group(1).strip()
            # Eliminar cualquier entidad de tiempo/fecha que ya hayamos detectado
            for entity_list in entities.values():
                for entity in entity_list:
                    if entity['value'].lower() in task_text:
                        task_text = task_text.replace(entity['value'].lower(), '').strip()
            
            if task_text:
                if 'task' not in entities:
                    entities['task'] = []
                entities['task'].append({
                    'value': task_text,
                    'label': 'task',
                    'groups': (task_text,)
                })
    
    # Si no se encontró tarea pero hay texto después de palabras clave
    if 'task' not in entities:
        task_patterns = [
            r'(?:necesito|quiero|deseo|me gustaría|podrías|puedes)\s+(?:recordar|agendar|programar|poner(?:me)?|hacer(?:me)?)(?:\s+un\s+recordatorio\s+para\s+|\s+que\s+|\s+de\s+|\s+el\s+|\s+la\s+|\s+algo\s+para\s+|\s+algo\s+el\s+|\s+algo\s+la\s+|\s+algo\s+los\s+|\s+algo\s+las\s+|\s+algo\s+un\s+|\s+algo\s+una\s+|\s+algo\s+unos\s+|\s+algo\s+unas\s+)?(.*?)(?:\.|\?|$)',
            r'(?:recuérdame|recordar|recordarme|recuerda|necesito recordar|agenda|agéndame|ponme|programa|programar)(?:\s+un\s+recordatorio\s+para\s+|\s+que\s+|\s+de\s+|\s+el\s+|\s+la\s+|\s+algo\s+para\s+|\s+algo\s+el\s+|\s+algo\s+la\s+|\s+algo\s+los\s+|\s+algo\s+las\s+|\s+algo\s+un\s+|\s+algo\s+una\s+|\s+algo\s+unos\s+|\s+algo\s+unas\s+)?(.*?)(?:\.|\?|$)'
        ]
        
        for pattern in task_patterns:
            task_match = re.search(pattern, text_lower, re.IGNORECASE)
            if task_match and task_match.group(1).strip():
                task_text = task_match.group(1).strip()
                # Eliminar cualquier entidad de tiempo/fecha que ya hayamos detectado
                for entity_list in entities.values():
                    for entity in entity_list:
                        if entity['value'].lower() in task_text:
                            task_text = task_text.replace(entity['value'].lower(), '').strip()
                
                if task_text:
                    if 'task' not in entities:
                        entities['task'] = []
                    entities['task'].append({
                        'value': task_text,
                        'label': 'task',
                        'groups': (task_text,)
                    })
                    break
    
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
    # Obtener el historial de la conversación
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    
    # Manejar la intención detectada
    lower_message = original_message.lower()
    
    # Manejar consultas sobre el clima
    if any(word in lower_message for word in [
        'hace frío', 'hace calor', 'qué temperatura', 'que temperatura',
        'cómo está el clima', 'como esta el clima', 'qué tiempo hace',
        'que tiempo hace', 'va a llover', 'hace sol', 'está nublado',
        'esta nublado', 'pronóstico', 'pronostico'
    ]):
        return "Actualmente no tengo acceso a información meteorológica en tiempo real. " \
               "¿Te gustaría que te ayude con algo más?"
    
    # Manejar consultas de fecha y hora
    if (intent == Intent.DATE_QUERY or 
        any(word in lower_message for word in [
            'qué día', 'que dia', 
            'qué fecha', 'que fecha', 
            'mañana', 'hoy', 
            'semana que viene',
            'qué hora es', 'que hora es',
            'dime la hora'
        ])):
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
        task = entities.get('task', '').strip()
        
        # Si no se especificó tarea, preguntar
        if not task or task.lower() in ['algo', 'algo?', 'algo.']:
            return "¿Qué te gustaría que te recuerde? Por ejemplo: 'Recuérdame comprar leche mañana'"
        
        time_info = entities.get('time', [{}])[0].get('value', '') if entities.get('time') else ''
        date_info = entities.get('date', [{}])[0].get('value', '') if entities.get('date') else ''
        
        # Crear el recordatorio
        reminder_id = str(uuid.uuid4())
        reminder = {
            "id": reminder_id,
            "task": task,
            "time": time_info,
            "date": date_info,
            "created_at": datetime.now().isoformat(),
            "session_id": session_id,
            "completed": False
        }
        
        # Guardar el recordatorio
        if 'reminders' not in globals():
            global reminders
            reminders = {}
        reminders[reminder_id] = reminder
        
        # Construir mensaje de confirmación
        message = f"✅ Perfecto, he creado un recordatorio para: {task}"
        if date_info:
            message += f" el {date_info}"
        if time_info:
            message += f" a las {time_info}"
            
        return message
    
    elif intent == Intent.LIST_REMINDERS:
        if 'reminders' not in globals() or not reminders:
            return "No tienes recordatorios programados. ¿Quieres crear uno?"
            
        user_reminders = [r for r in reminders.values() if r.get('session_id') == session_id]
        if not user_reminders:
            return "No tienes recordatorios programados. ¿Quieres crear uno?"
        
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

def parse_reminder(message: str) -> Optional[Tuple[str, datetime]]:
    """
    Intenta extraer información de recordatorio del mensaje.
    Ejemplos:
    - "Recuérdame comprar leche mañana a las 5pm"
    - "Agenda una reunión el lunes a las 10am"
    """
    # Expresiones regulares para fechas y horas
    time_patterns = [
        (r'(?:a las|a la|al|las|los)\s+(\d{1,2})(?::(\d{2}))?\s*(?:hs|h|horas|:)?\s*(am|pm)?', 'time'),
        (r'(mañana|pasado mañana|hoy)', 'relative_day'),
        (r'(lunes|martes|miércoles|jueves|viernes|sábado|domingo)', 'weekday'),
        (r'(próxima semana|siguiente semana)', 'next_week'),
    ]
    
    # Extraer el texto del recordatorio
    reminder_text = message.lower()
    reminder_text = re.sub(r'^(?:recuérdame|recuerda|agenda|programa|pon|agendame|programame|necesito recordar|quiero que me recuerdes|quiero recordar|deseo recordar|me gustaría recordar|por favor recuérdame|por favor recuerda|por favor agenda|por favor programa|por favor pon|por favor agendame|por favor programame|por favor necesito recordar|por favor quiero que me recuerdes|por favor quiero recordar|por favor deseo recordar|por favor me gustaría recordar|por favor recuérdame que|por favor recuerda que|por favor agenda que|por favor programa que|por favor pon que|por favor agendame que|por favor programame que|por favor necesito recordar que|por favor quiero que me recuerdes que|por favor quiero recordar que|por favor deseo recordar que|por favor me gustaría recordar que)', '', reminder_text).strip()
    
    # Por ahora, devolvemos un recordatorio simple
    # En una implementación real, aquí analizaríamos la fecha/hora
    if 'recordar' in message.lower() or 'recordatorio' in message.lower() or 'recordarme' in message.lower():
        # Intentar extraer la fecha/hora
        time_info = "próximamente"
        return (f"He creado un recordatorio para: {reminder_text} {time_info}", datetime.now() + timedelta(hours=1))
    
    return None

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

@router.post("/chat/")
async def chat_endpoint(
    message: str = Form(...),
    session_id: str = Form("default")
):
    """
    Endpoint principal para el chat.
    Procesa el mensaje del usuario, detecta la intención, extrae entidades
    y genera una respuesta.
    """
    try:
        print(f"Mensaje recibido: {message}")
        print(f"Session ID: {session_id}")
        
        # Inicializar el historial de la sesión si no existe
        if session_id not in chat_histories:
            chat_histories[session_id] = []
        
        # Detectar la intención y extraer entidades
        intent, entities = detect_intent(message)
        logger.info(f"Intención detectada: {intent}") # Usar logger en lugar de print
        logger.info(f"Entidades extraídas: {json.dumps(entities, indent=2, ensure_ascii=False)}") # Usar logger en lugar de print
        
        # Generar respuesta basada en la intención
        response_text = generate_response(intent, entities, session_id, message)
        
        # Crear mensajes para el historial
        user_message = Message(
            role=MessageRole.USER,
            content=message
        )
        
        assistant_message = Message(
            role=MessageRole.ASSISTANT,
            content=response_text
        )
        
        # Agregar mensajes al historial
        chat_histories[session_id].extend([user_message, assistant_message])
        
        # Limitar el historial a los últimos 10 mensajes
        chat_histories[session_id] = chat_histories[session_id][-10:]
        
        # Manejar intenciones específicas (mover lógica de recordatorio aquí si es necesario)
        # La lógica de creación de recordatorios ya está en generate_response y actualiza 'reminders'
        # Solo necesitamos asegurarnos de que el estado 'notified' se añada al crear.
        # Revisando generate_response, ya añade "notified": False.

        # Llamar a la función de verificación de recordatorios (para demostración)
        check_and_notify_reminders()

        return {
            "response": response_text,
            "status": "success",
            "intent": intent.value,
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
