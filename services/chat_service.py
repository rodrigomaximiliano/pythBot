"""
Servicio de chat que maneja la lógica de procesamiento de mensajes.
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import random
from dateutil import parser # Importar dateutil.parser

from config.chat_responses import get_random_response, INTENT_KEYWORDS
from nlp.intent_classifier import process_message as process_nlp_message # Importar la función de NLP
from nlp.response_generator import response_generator # Importar la instancia global del generador de respuestas
from utils.recordatorios import agregar_recordatorio, obtener_recordatorios # Importar funciones de recordatorios
from utils.eventos import agregar_evento, obtener_eventos # Importar funciones de eventos

from utils.intent_recognizer import IntentRecognizer, IntentType as IntentRecognizerType

logger = logging.getLogger(__name__)

class ChatService:
    """Servicio para manejar la lógica del chat."""
    
    def __init__(self):
        self.chat_histories: Dict[str, List[Dict[str, Any]]] = {}
        self.intent_recognizer = IntentRecognizer() # Instanciar el reconocedor de intenciones
        # No necesitamos instanciar ResponseGenerator aquí porque importamos la instancia global
    
    async def process_message(self, message: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Procesa un mensaje del usuario y devuelve una respuesta.
        
        Args:
            message: El mensaje del usuario
            session_id: Identificador único de la sesión
            
        Returns:
            Dict con la respuesta y metadatos
        """
        # Inicializar historial si no existe
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = []
        
        # Agregar mensaje al historial
        user_message = {
            "role": "user",
            "content": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.chat_histories[session_id].append(user_message)
        
        try:
            # Determine intent type first (using regex or NLP fallback)
            recognized_intent_type, _ = self.intent_recognizer.recognize(message)

            # Import process_intent and IntentType from utils.command_handlers
            from utils.command_handlers import process_intent, IntentType

            if recognized_intent_type != IntentRecognizerType.UNKNOWN:
                # Use regex recognized intent type
                try:
                    intent_type = IntentType[recognized_intent_type.name]
                except KeyError:
                    intent_type = IntentType.UNKNOWN
                logger.info(f"Intención reconocida por regex: {intent_type.name}")
            else:
                # Fallback to embedding classification for intent type
                intent_str_nlp, _, _ = process_nlp_message(message) # Only get intent string from NLP
                try:
                    intent_type = IntentType[intent_str_nlp.upper()]
                except KeyError:
                    intent_type = IntentType.UNKNOWN
                logger.info(f"Intención detectada por NLP (solo tipo): {intent_type.name}")

            # Now, use the determined intent_type to extract entities using the NLP engine's robust logic
            # We call process_nlp_message again, but this time we care about entities and confidence
            # Note: The current process_nlp_message returns intent_str, entities, confidence.
            # We already determined intent_type, so we'll primarily use the entities and confidence from this call.
            _, entities, confidence = process_nlp_message(message)
            logger.info(f"Entidades extraídas por NLP (usando intención {intent_type.name}): {entities}")
            logger.info(f"Confianza de NLP: {confidence}")

            # Process the determined intent_type and extracted entities
            if intent_type == IntentType.UNKNOWN:
                # Handle unknown intent using the response generator as fallback
                chat_history_context = [msg["content"] for msg in self.chat_histories.get(session_id, [])]
                response_text = response_generator.generate_response(message, context=chat_history_context)
            else:
                # Process known intent using command handlers
                # Pass the entities extracted by the NLP engine
                response_text = process_intent(intent_type, entities, session_id)

            # The response from process_intent is just the text, we need to structure it
            response_data = {
                "response": response_text,
                "suggestions": [], # Handlers could return suggestions if needed
                "intent": intent_type.name.lower(), # Use the determined intent type name
                "confidence": confidence, # Add the confidence from NLP
                "entities": entities # Add the entities from NLP
            }

        except Exception as e:
            logger.error(f"Error al procesar mensaje: {e}", exc_info=True)
            response_data = get_random_response("unknown") # Respuesta genérica de error

        # Agregar respuesta al historial
        bot_message = {
            "role": "assistant",
            "content": response_data["response"],
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {
                "intent": response_data.get("intent", "unknown"), # Usar la intención del response_data
                "confidence": response_data.get("confidence", 0.0), # Agregar la confianza (si está disponible)
                "entities": response_data.get("entities", {}), # Agregar las entidades (si están disponibles)
                "suggestions": response_data.get("suggestions", [])
            }
        }
        self.chat_histories[session_id].append(bot_message)

        # Limitar el historial
        self._limit_history(session_id)

        return response_data
