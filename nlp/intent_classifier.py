from typing import Dict, List, Tuple, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import logging
from dateutil import parser # Importar dateutil.parser
import parsedatetime as pdt # Importar parsedatetime
import time # Importar time
from datetime import datetime # Importar datetime para usarlo en extract_entities

logger = logging.getLogger(__name__)

class NLUEngine:
    """Motor de Procesamiento de Lenguaje Natural para el chatbot."""
    
    def __init__(self):
        # Inicializar el parser de parsedatetime
        self.cal = pdt.Calendar()
        # Cargar el modelo multilingüe de Sentence Transformers
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        
        # Definir las intenciones y ejemplos
        self.intents = {
            "greeting": [
                "Hola", "Buenos días", "Buenas tardes", "Buenas noches",
                "¿Cómo estás?", "¿Qué tal?", "¿Cómo te va?", "Hola, ¿cómo estás?",
                "¿Cómo andas?", "¿Qué hay de nuevo?", "Hola, ¿qué tal todo?", "¿Qué onda?",
                "Saludos", "Qué onda", "Buen día", "Buenas", "Hola, qué haces" # Nuevos ejemplos
            ],
            "farewell": [
                "Adiós", "Hasta luego", "Nos vemos", "Chao", "Hasta pronto",
                "Hasta la próxima", "Me despido", "Hasta mañana", "Hasta la vista",
                "Nos vemos luego", "Hasta otro día", "Adiós, que te vaya bien", "Chao pescao",
                "Me piro", "Hasta la vista baby" # Nuevos ejemplos
            ],
            "thanks": [
                "Gracias", "Muchas gracias", "Te lo agradezco", "Mil gracias",
                "Te agradezco", "Gracias por tu ayuda", "Te lo agradezco mucho",
                "Muy amable", "Te lo agradezco de antemano", "Gracias por todo",
                "Gracias mil", "Gracias totales", "Gracias por la info" # Nuevos ejemplos
            ],
            "help": [
                "Ayuda", "¿Qué puedes hacer?", "¿Para qué sirves?", "¿Cómo te llamas?",
                "¿Qué sabes hacer?", "¿Cómo funcionas?", "¿Qué comandos tienes?",
                "¿Me puedes ayudar?", "¿Qué haces?", "¿Cuáles son tus funciones?",
                "¿Cómo te puedo usar?", "¿Cómo trabajas?", "Necesito ayuda",
                "Explícame qué haces", "Qué funciones tienes", "Cómo te uso" # Nuevos ejemplos
            ],
            "create_reminder": [
                "Recuérdame comprar leche", "Agenda una reunión mañana",
                "Necesito recordar llamar al médico", "Ponme un recordatorio para el dentista",
                "Quiero que me recuerdes algo", "Puedes agendar una cita",
                "Necesito que me recuerdes hacer la tarea", "Agéndame una llamada con Juan",
                "Recuérdame que tengo que ir al banco", "Quiero que me avises mañana temprano",
                "Pon una alarma para las 3pm", "Recuérdame mi reunión del viernes",
                "Necesito un recordatorio para el cumpleaños de mamá", "Crear recordatorio: pagar facturas el lunes",
                "Agendar: ir al gimnasio a las 7pm", "Recordarme: enviar email a Pedro pasado mañana",
                "Pon un recordatorio para ir al supermercado", "Necesito que me recuerdes recoger el paquete" # Nuevos ejemplos
            ],
            "list_reminders": [
                "¿Qué recordatorios tengo?", "Muestra mis recordatorios",
                "Lista de recordatorios", "¿Qué tengo agendado?",
                "¿Qué tengo pendiente?", "Muéstrame mis recordatorios",
                "¿Qué recordatorios hay?", "¿Tengo algo programado?",
                "Dime mis recordatorios", "¿Qué tengo en mi agenda?",
                "Ver mis recordatorios", "Cuáles son mis recordatorios",
                "Lista de mis pendientes", "Qué recordatorios tengo para hoy" # Nuevos ejemplos
            ]
        }
        
        # Generar embeddings para los ejemplos
        self._generate_embeddings()
    
    def _generate_embeddings(self):
        """Genera los embeddings para los ejemplos de entrenamiento."""
        self.embeddings = {}
        for intent, examples in self.intents.items():
            # Generar embeddings para todos los ejemplos de esta intención
            intent_embeddings = self.model.encode(examples, convert_to_tensor=True, device=self.device)
            # Calcular el embedding promedio para esta intención
            self.embeddings[intent] = torch.mean(intent_embeddings, dim=0)
    
    def predict_intent(self, text: str, threshold: float = 0.5) -> Tuple[str, float]:
        """
        Predice la intención del texto de entrada.
        
        Args:
            text: Texto del usuario
            threshold: Umbral de confianza mínimo para considerar una predicción válida
            
        Returns:
            Tupla con (intención_predicha, puntuación_de_confianza)
        """
        # Generar embedding para el texto de entrada
        text_embedding = self.model.encode(text, convert_to_tensor=True, device=self.device)
        
        # Calcular similitud con cada intención
        similarities = {}
        for intent, intent_embedding in self.embeddings.items():
            # Calcular similitud del coseno
            similarity = torch.nn.functional.cosine_similarity(
                text_embedding.unsqueeze(0),
                intent_embedding.unsqueeze(0)
            ).item()
            similarities[intent] = similarity
        
        # Obtener la intención con mayor similitud
        predicted_intent = max(similarities.items(), key=lambda x: x[1])
        
        # Aplicar umbral de confianza
        if predicted_intent[1] < threshold:
            return "unknown", predicted_intent[1]
            
        return predicted_intent
    
    def extract_entities(self, text: str, intent: str) -> Dict[str, Any]:
        """
        Extrae entidades relevantes del texto según la intención.
        
        Args:
            text: Texto del usuario
            intent: Intención detectada
            
        Returns:
            Diccionario con las entidades extraídas
        """
        entities = {}
        
        # Extraer entidades según la intención
        if intent == "create_reminder":
            # Lista de verbos y frases que indican una tarea
            task_indicators = [
                "recuérdame", "recordar", "recordarme", "recuerda", "necesito recordar",
                "agenda", "agéndame", "ponme", "programa", "programar", "necesito",
                "quiero", "deseo", "me gustaría", "podrías", "puedes", "tengo que",
                "debo", "tendría que"
            ]
            
            # Buscar la tarea después de los indicadores
            task = None
            for indicator in task_indicators:
                if indicator in text.lower():
                    # Extraer todo después del indicador
                    task_part = text[text.lower().find(indicator) + len(indicator):].strip()
                    if task_part:
                        task = task_part
                        break
            
            # Si no se encontró un indicador claro, usar todo el texto como tarea
            if task is None:
                task = text

            # Extraer fechas y horas usando parsedatetime
            date_time_entity = None
            try:
                # Obtener la hora actual para el contexto del parseo
                current_time = time.localtime()
                # Parsear el texto
                date_time_tuple, parse_status = self.cal.parseDT(datetimeString=text, tzinfo=None, sourceTime=current_time)
                
                if parse_status != 0: # parse_status > 0 indica que se encontró una fecha/hora
                    # Devolver el objeto datetime directamente
                    date_time_entity = date_time_tuple
                    entities['date_time'] = date_time_entity
                    logger.info(f"Fecha/Hora extraída con parsedatetime: {entities['date_time']}")
                else:
                     logger.info("No se pudo extraer fecha/hora con parsedatetime.")
                     # Eliminamos el fallback a dateutil.parser para usar consistentemente parsedatetime

            except Exception as e:
                logger.error(f"Error al extraer fecha/hora con parsedatetime: {e}")

            # Refinar la tarea si se extrajo una entidad de fecha/hora
            # Ahora date_time_entity es un objeto datetime o None
            if date_time_entity is not None and task:
                # Intentar encontrar la posición de la fecha/hora en el texto original
                # Esto es una aproximación y puede no ser perfecta
                # Como ahora tenemos un objeto datetime, podemos intentar formatearlo para buscar en el texto
                date_time_str_in_text = None
                # Considerar algunos formatos comunes que parsedatetime podría haber interpretado
                for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%H:%M:%S", "%I:%M %p"]:
                    try:
                        formatted_dt_str = date_time_entity.strftime(fmt)
                        if formatted_dt_str in text:
                            date_time_str_in_text = formatted_dt_str
                            break
                    except ValueError:
                        pass # Ignorar si el formato no coincide

                if date_time_str_in_text:
                    # Si encontramos la cadena de fecha/hora en el texto original,
                    # asumimos que la tarea es la parte antes de eso.
                    task_before_date = text.split(date_time_str_in_text, 1)[0].strip()
                    # Asegurarnos de que la tarea refinada no esté vacía
                    if task_before_date:
                        entities['task'] = task_before_date
                    else:
                        # Si la parte antes de la fecha está vacía, mantenemos la tarea original
                        entities['task'] = task
                else:
                     # Si no pudimos encontrar la cadena de fecha/hora en el texto,
                     # mantenemos la tarea original extraída después del indicador.
                     entities['task'] = task
            else:
                 # Si no hay entidad de fecha/hora, o no se encontró tarea inicialmente,
                 # mantenemos la tarea original (que podría ser todo el texto si no hubo indicador).
                 entities['task'] = task

        elif intent == "create_event":
            # Para eventos, intentamos extraer título, ubicación y descripción
            # Esto es una extracción básica y puede necesitar mejoras
            
            # Extraer título (podríamos asumir que es la parte principal antes de la fecha/hora)
            # Por ahora, usaremos todo el texto como título si no se extrae otra cosa
            entities['title'] = text
            
            # Extraer ubicación (ejemplo básico: buscar "en [lugar]")
            location_indicators = [" en "]
            for indicator in location_indicators:
                if indicator in text.lower():
                    parts = text.lower().split(indicator, 1)
                    if len(parts) > 1:
                        # Asumir que la ubicación es lo que sigue al indicador
                        entities['ubicacion'] = text[text.lower().find(indicator) + len(indicator):].strip()
                        # Podríamos refinar esto para no incluir la fecha/hora si está al final
                        break
                        
            # Extraer descripción (ejemplo básico: buscar "sobre [tema]" o "para [propósito]")
            description_indicators = [" sobre ", " para "]
            for indicator in description_indicators:
                 if indicator in text.lower():
                    parts = text.lower().split(indicator, 1)
                    if len(parts) > 1:
                        # Asumir que la descripción es lo que sigue al indicador
                        entities['descripcion'] = text[text.lower().find(indicator) + len(indicator):].strip()
                        # Podríamos refinar esto
                        break
 
            # Extraer fechas y horas usando parsedatetime (aplicable a recordatorios y eventos)
            # Ya se hizo la extracción de fecha/hora arriba, pero la repetimos aquí por claridad
            # en el contexto de create_event, aunque idealmente se haría una sola vez.
            # Para evitar duplicación, podríamos mover la extracción de fecha/hora fuera de los if de intención.
            try:
                # Obtener la hora actual para el contexto del parseo
                current_time = time.localtime()
                # Parsear el texto
                date_time_tuple, parse_status = self.cal.parseDT(datetimeString=text, tzinfo=None, sourceTime=current_time)
                
                if parse_status != 0: # parse_status > 0 indica que se encontró una fecha/hora
                    # Devolver el objeto datetime directamente
                    entities['date_time'] = date_time_tuple
                    logger.info(f"Fecha/Hora extraída con parsedatetime: {entities['date_time']}")
                else:
                     logger.info("No se pudo extraer fecha/hora con parsedatetime.")
                     # Eliminamos el fallback a dateutil.parser
 
            except Exception as e:
                logger.error(f"Error al extraer fecha/hora con parsedatetime: {e}")

        # La extracción de fecha/hora se movió fuera de los if de intención para evitar duplicación
        # y asegurar que se intente para cualquier intención donde pueda ser relevante.
        # Sin embargo, la lógica actual la tiene dentro de cada if.
        # Para esta mejora, la mantendré dentro del if de create_reminder y create_event
        # para no alterar demasiado la estructura existente, pero es un área de mejora futura.

        return entities
    

# Instancia global del motor de NLP
nlp_engine = NLUEngine()

def process_message(text: str) -> Tuple[str, Dict[str, Any], float]:
    """
    Procesa un mensaje del usuario y devuelve la intención y entidades.
    
    Args:
        text: Mensaje del usuario
        
    Returns:
        Tupla con (intención, entidades, confianza)
    """
    # Detectar intención
    intent, confidence = nlp_engine.predict_intent(text)
    
    # Extraer entidades relevantes
    entities = nlp_engine.extract_entities(text, intent)
    
    return intent, entities, confidence
