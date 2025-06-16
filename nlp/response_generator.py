import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ResponseGenerator:
    def __init__(self, model_name="microsoft/DialoGPT-small"):
        """
        Inicializa el generador de respuestas con un modelo de lenguaje.
        Usa DialoGPT-small por defecto, que está optimizado para diálogos.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Cargando modelo de lenguaje en {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            self.model.eval()
            logger.info("Modelo de lenguaje cargado exitosamente")
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {e}")
            raise
    
    def generate_response(self, message: str, context: List[str] = None, max_length: int = 100) -> str:
        """
        Genera una respuesta basada en el mensaje del usuario y el contexto de la conversación.
        
        Args:
            message: El mensaje del usuario
            context: Lista de mensajes previos en la conversación
            max_length: Longitud máxima de la respuesta generada
            
        Returns:
            str: La respuesta generada
        """
        try:
            # Preprocesar el contexto
            if context is None:
                context = []
                
            # Limitar el historial para no exceder el tamaño máximo del modelo
            context = context[-3:]  # Tomar solo los últimos 3 mensajes
            
            # Crear el prompt con el historial de la conversación
            prompt = "\n".join(context + [f"Usuario: {message}", "Bot:"])
            
            # Codificar el prompt
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generar respuesta
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True
                )
            
            # Decodificar la respuesta
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extraer solo la parte de la respuesta del bot
            response = response.split("Bot:")[-1].strip()
            
            # Limpiar la respuesta
            response = self._clean_response(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error al generar respuesta: {e}")
            return "Lo siento, tuve un problema al procesar tu mensaje. ¿Podrías intentarlo de nuevo?"
    
    def _clean_response(self, text: str) -> str:
        """Limpia la respuesta generada."""
        # Eliminar múltiples espacios
        text = " ".join(text.split())
        # Eliminar repeticiones
        if "..." in text:
            text = text.split("...")[0] + "..."
        # Capitalizar la primera letra
        if text:
            text = text[0].upper() + text[1:]
        return text

# Instancia global
response_generator = ResponseGenerator()
