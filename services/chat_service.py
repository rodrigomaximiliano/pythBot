"""
Servicio de chat que maneja la lógica de procesamiento de mensajes.
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import random

from config.chat_responses import get_random_response, INTENT_KEYWORDS

logger = logging.getLogger(__name__)

class ChatService:
    """Servicio para manejar la lógica del chat."""
    
    def __init__(self):
        self.chat_histories: Dict[str, List[Dict[str, Any]]] = {}
    
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
            # Detectar intención
            intent = self._detect_intent(message)
            
            # Procesar según la intención
            if intent == "create_reminder":
                response = self._handle_create_reminder(message, session_id)
            elif intent == "list_reminders":
                response = self._handle_list_reminders(session_id)
            elif intent == "create_event":
                response = self._handle_create_event(message, session_id)
            elif intent == "list_events":
                response = self._handle_list_events(session_id)
            else:
                # Para saludos, despedidas, ayuda, etc.
                response = get_random_response(intent)
                
        except Exception as e:
            logger.error(f"Error al procesar mensaje: {e}", exc_info=True)
            response = get_random_response("unknown")
        
        # Agregar respuesta al historial
        bot_message = {
            "role": "assistant",
            "content": response["response"],
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {
                "intent": response.get("intent", "unknown"),
                "suggestions": response.get("suggestions", [])
            }
        }
        self.chat_histories[session_id].append(bot_message)
        
        # Limitar el historial
        self._limit_history(session_id)
        
        return response
    
    def _detect_intent(self, text: str) -> str:
        """Detecta la intención del usuario basado en el texto."""
        if not text:
            return "unknown"
            
        text = text.lower()
        
        # Buscar coincidencias de palabras clave
        for intent, keywords in INTENT_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                return intent
                
        # Si no se detecta ninguna intención específica
        return "unknown"
    
    def _handle_create_reminder(self, text: str, session_id: str) -> Dict[str, Any]:
        """Maneja la creación de un recordatorio."""
        # Aquí iría la lógica para extraer fecha y texto del recordatorio
        # Por ahora, usamos un placeholder
        reminder_text = text  # Esto debería ser procesado por NLP
        reminder_date = "mañana"  # Esto debería ser extraído del texto
        
        # Aquí iría el código para guardar el recordatorio
        # save_reminder(reminder_text, reminder_date, session_id)
        
        return get_random_response(
            "reminder_created",
            text=reminder_text,
            date=reminder_date
        )
    
    def _handle_list_reminders(self, session_id: str) -> Dict[str, Any]:
        """Maneja la lista de recordatorios."""
        # Aquí iría el código para obtener los recordatorios
        # reminders = get_reminders(session_id)
        reminders = []  # Placeholder
        
        if not reminders:
            return get_random_response("no_reminders")
            
        # Formatear la lista de recordatorios
        reminders_list = "\n".join([f"- {r['text']} ({r['date']})" for r in reminders])
        return {
            "response": f"Tus recordatorios son:\n{reminders_list}",
            "suggestions": ["Crear recordatorio", "Eliminar recordatorio"],
            "intent": "list_reminders"
        }
    
    def _handle_create_event(self, text: str, session_id: str) -> Dict[str, Any]:
        """Maneja la creación de un evento."""
        # Aquí iría la lógica para extraer detalles del evento
        # Por ahora, usamos placeholders
        event_title = "Evento"  # Debería extraerse del texto
        event_date = "próxima semana"  # Debería extraerse del texto
        
        # Aquí iría el código para guardar el evento
        # save_event(event_title, event_date, session_id)
        
        return get_random_response(
            "event_created",
            title=event_title,
            date=event_date
        )
    
    def _handle_list_events(self, session_id: str) -> Dict[str, Any]:
        """Maneja la lista de eventos."""
        # Aquí iría el código para obtener los eventos
        # events = get_events(session_id)
        events = []  # Placeholder
        
        if not events:
            return get_random_response("no_events")
            
        # Formatear la lista de eventos
        events_list = "\n".join([f"- {e['title']} ({e['date']})" for e in events])
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
