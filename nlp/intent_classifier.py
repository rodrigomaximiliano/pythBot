from typing import Dict, List, Tuple, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import logging

logger = logging.getLogger(__name__)

class NLUEngine:
    """Motor de Procesamiento de Lenguaje Natural para el chatbot."""
    
    def __init__(self):
        # Cargar el modelo multilingüe de Sentence Transformers
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        
        # Definir las intenciones y ejemplos
        self.intents = {
            "greeting": [
                "Hola", "Buenos días", "Buenas tardes", "Buenas noches",
                "¿Cómo estás?", "¿Qué tal?", "¿Cómo te va?", "Hola, ¿cómo estás?",
                "¿Cómo andas?", "¿Qué hay de nuevo?", "Hola, ¿qué tal todo?", "¿Qué onda?"
            ],
            "farewell": [
                "Adiós", "Hasta luego", "Nos vemos", "Chao", "Hasta pronto",
                "Hasta la próxima", "Me despido", "Hasta mañana", "Hasta la vista",
                "Nos vemos luego", "Hasta otro día"
            ],
            "thanks": [
                "Gracias", "Muchas gracias", "Te lo agradezco", "Mil gracias",
                "Te agradezco", "Gracias por tu ayuda", "Te lo agradezco mucho",
                "Muy amable", "Te lo agradezco de antemano"
            ],
            "help": [
                "Ayuda", "¿Qué puedes hacer?", "¿Para qué sirves?", "¿Cómo te llamas?",
                "¿Qué sabes hacer?", "¿Cómo funcionas?", "¿Qué comandos tienes?",
                "¿Me puedes ayudar?", "¿Qué haces?", "¿Cuáles son tus funciones?",
                "¿Cómo te puedo usar?", "¿Cómo trabajas?"
            ],
            "create_reminder": [
                "Recuérdame comprar leche", "Agenda una reunión mañana",
                "Necesito recordar llamar al médico", "Ponme un recordatorio para el dentista",
                "Quiero que me recuerdes algo", "Puedes agendar una cita",
                "Necesito que me recuerdes hacer la tarea", "Agéndame una llamada con Juan",
                "Recuérdame que tengo que ir al banco", "Quiero que me avises mañana temprano",
                "Pon una alarma para las 3pm", "Recuérdame mi reunión del viernes",
                "Necesito un recordatorio para el cumpleaños de mamá"
            ],
            "list_reminders": [
                "¿Qué recordatorios tengo?", "Muestra mis recordatorios",
                "Lista de recordatorios", "¿Qué tengo agendado?",
                "¿Qué tengo pendiente?", "Muéstrame mis recordatorios",
                "¿Qué recordatorios hay?", "¿Tengo algo programado?",
                "Dime mis recordatorios", "¿Qué tengo en mi agenda?"
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
        
        # Extraer tareas para recordatorios
        if intent == "create_reminder":
            # Lista de verbos y frases que indican una tarea
            task_indicators = [
                "recuérdame", "recordar", "recordarme", "recuerda", "necesito recordar",
                "agenda", "agéndame", "ponme", "programa", "programar", "necesito",
                "quiero", "deseo", "me gustaría", "podrías", "puedes", "tengo que",
                "debo", "tendría que"
            ]
            
            # Buscar la tarea después de los indicadores
            for indicator in task_indicators:
                if indicator in text.lower():
                    # Extraer todo después del indicador
                    task = text[text.lower().find(indicator) + len(indicator):].strip()
                    if task:
                        entities['task'] = task
                        break
            
            # Si no se encontró un indicador claro, usar todo el texto como tarea
            if 'task' not in entities:
                entities['task'] = text
        
        # Extraer fechas y horas (puedes expandir esto según sea necesario)
        # Aquí iría la lógica para extraer fechas y horas usando expresiones regulares
        
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
