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

logger = logging.getLogger(__name__)

class ChatService:
    """Servicio para manejar la lógica del chat."""
    
    def __init__(self):
        self.chat_histories: Dict[str, List[Dict[str, Any]]] = {}
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
            # Procesar mensaje con el motor de NLP
            intent, entities, confidence = process_nlp_message(message)
            logger.info(f"Intención detectada: {intent} con confianza {confidence}")
            logger.info(f"Entidades extraídas: {entities}")
            
            # Procesar según la intención detectada
            if intent == "create_reminder":
                # Pasar las entidades extraídas al manejador
                response = self._handle_create_reminder(entities, session_id)
            elif intent == "list_reminders":
                response = self._handle_list_reminders(session_id)
            elif intent == "create_event":
                # Pasar las entidades extraídas al manejador
                response = self._handle_create_event(entities, session_id)
            elif intent == "list_events":
                response_data = self._handle_list_events(session_id)
            else:
                # Para intenciones generales o desconocidas, usar el generador de respuestas
                # Obtener el historial de chat para el contexto
                chat_history_context = [msg["content"] for msg in self.chat_histories.get(session_id, [])]
                
                # Generar respuesta usando el modelo de lenguaje
                generated_text = response_generator.generate_response(message, context=chat_history_context)
                
                response_data = {
                    "response": generated_text,
                    "suggestions": [], # Puedes agregar sugerencias relevantes aquí si es necesario
                    "intent": intent # Mantener la intención detectada por el NLP
                }
                
        except Exception as e:
            logger.error(f"Error al procesar mensaje: {e}", exc_info=True)
            response_data = get_random_response("unknown")
        
        # Agregar respuesta al historial
        bot_message = {
            "role": "assistant",
            "content": response_data["response"],
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {
                "intent": intent, # Usar la intención detectada por el NLP
                "confidence": confidence, # Agregar la confianza
                "entities": entities, # Agregar las entidades
                "suggestions": response_data.get("suggestions", [])
            }
        }
        self.chat_histories[session_id].append(bot_message)
        
        # Limitar el historial
        self._limit_history(session_id)
        
        return response_data
    
    # Eliminamos el método _detect_intent ya que no se usará más
    # def _detect_intent(self, text: str) -> str:
    #     """Detecta la intención del usuario basado en el texto."""
    #     if not text:
    #         return "unknown"
            
    #     text = text.lower()
        
    #     # Buscar coincidencias de palabras clave
    #     for intent, keywords in INTENT_KEYWORDS.items():
    #         if any(keyword in text for keyword in keywords):
    #             return intent
                
    #     # Si no se detecta ninguna intención específica
    #     return "unknown"
    
    def _handle_create_reminder(self, entities: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Maneja la creación de un recordatorio."""
        # Ahora usamos las entidades extraídas por el NLP
        reminder_text = entities.get('task', 'algo') # Usar la entidad 'task' si existe
        date_time_str = entities.get('date_time') # Obtener la cadena de fecha/hora

        if not date_time_str:
            # Si no se extrajo fecha/hora, pedir más información o usar un valor por defecto
            return get_random_response("clarify_datetime") # Asumiendo que existe una respuesta para esto
            
        try:
            # Convertir la cadena de fecha/hora a un objeto datetime
            reminder_date_time = parser.parse(date_time_str)
            
            # Agregar el recordatorio usando la función importada
            success, message = agregar_recordatorio(session_id, reminder_text, reminder_date_time)
            
            if success:
                return get_random_response(
                    "reminder_created",
                    text=reminder_text,
                    date=message.split(" para ")[-1] # Extraer la fecha formateada del mensaje de éxito
                )
            else:
                return {"response": f"No pude crear el recordatorio: {message}", "suggestions": [], "intent": "create_reminder_failed"}
                
        except parser.ParserError:
            return get_random_response("clarify_datetime") # Pedir clarificación si la fecha/hora no es válida
        except Exception as e:
            logger.error(f"Error al crear recordatorio: {e}", exc_info=True)
            return get_random_response("error_creating_reminder") # Asumiendo que existe una respuesta para esto
    
    def _handle_list_reminders(self, session_id: str) -> Dict[str, Any]:
        """Maneja la lista de recordatorios."""
        # Obtener los recordatorios usando la función importada
        reminders = obtener_recordatorios(session_id)
        
        if not reminders:
            return get_random_response("no_reminders")
            
        # Formatear la lista de recordatorios
        # La función __str__ de Recordatorio ya formatea la salida
        reminders_list = "\n".join([str(r) for r in reminders])
        return {
            "response": f"Tus recordatorios son:\n{reminders_list}",
            "suggestions": ["Crear recordatorio", "Eliminar recordatorio"],
            "intent": "list_reminders"
        }
    
    def _handle_create_event(self, entities: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Maneja la creación de un evento."""
        # Ahora usamos las entidades extraídas por el NLP
        event_title = entities.get('task', entities.get('title', 'un evento')) # Usar 'task' o 'title'
        date_time_str = entities.get('date_time') # Obtener la cadena de fecha/hora
        ubicacion = entities.get('ubicacion') # Asumiendo que también podríamos extraer ubicación
        descripcion = entities.get('descripcion') # Asumiendo que también podríamos extraer descripción

        if not date_time_str:
            # Si no se extrajo fecha/hora, pedir más información o usar un valor por defecto
            return get_random_response("clarify_datetime") # Reutilizamos la respuesta de recordatorios
            
        try:
            # Convertir la cadena de fecha/hora a un objeto datetime
            event_date_time = parser.parse(date_time_str)
            
            # Agregar el evento usando la función importada
            success, message = agregar_evento(session_id, event_title, event_date_time, ubicacion, descripcion)
            
            if success:
                # El mensaje de éxito de agregar_evento ya incluye el título y la fecha formateada
                return {"response": message, "suggestions": [], "intent": "event_created"}
            else:
                return {"response": f"No pude crear el evento: {message}", "suggestions": [], "intent": "create_event_failed"}
                
        except parser.ParserError:
            return get_random_response("clarify_datetime") # Pedir clarificación si la fecha/hora no es válida
        except Exception as e:
            logger.error(f"Error al crear evento: {e}", exc_info=True)
            return get_random_response("error_creating_event") # Asumiendo que existe una respuesta para esto
    
    def _handle_list_events(self, session_id: str) -> Dict[str, Any]:
        """Maneja la lista de eventos."""
        # Obtener los eventos usando la función importada
        events = obtener_eventos(session_id)
        
        if not events:
            return get_random_response("no_events")
            
        # Formatear la lista de eventos
        # La función __str__ de Evento ya formatea la salida
        events_list = "\n".join([str(e) for e in events])
        return {
            "response": f"Tus eventos son:\n{events_list}",
            "suggestions": ["Crear evento", "Eliminar evento"],
            "intent": "list_events"
        }
    
    def _limit_history(self, session_id: str, max_messages: int = 20) -> None:
        """Limita el tamaño del historial de mensajes."""
        if session_id in self.chat_histories:
            self.chat_histories[session_id] = self.chat_histories[session_id][-max_messages:]
    
    def get_chat_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Obtiene el historial de chat para una sesión."""
        return self.chat_histories.get(session_id, [])
    
    def clear_history(self, session_id: str) -> None:
        """Borra el historial de una sesión."""
        if session_id in self.chat_histories:
            del self.chat_histories[session_id]

# Instancia global del servicio de chat
chat_service = ChatService()
