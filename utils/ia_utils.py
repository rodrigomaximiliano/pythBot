import torch
from typing import Any, List, Tuple, Optional
import logging
import re

def limpiar_respuesta(respuesta: str) -> str:
    """
    Limpia la respuesta generada por el modelo, eliminando espacios, saltos de línea y frases irrelevantes.
    """
    if not respuesta:
        return ""
    # Elimina espacios y saltos de línea redundantes
    respuesta = respuesta.strip().replace('\n', ' ').replace('\r', ' ')
    # Elimina repeticiones de palabras vacías típicas de modelos
    respuesta = re.sub(r'\b(um+|eh+|ah+)\b', '', respuesta, flags=re.IGNORECASE)
    # Elimina dobles espacios
    respuesta = re.sub(r'\s+', ' ', respuesta)
    return respuesta.strip()

def es_respuesta_no_entendida(respuesta: str) -> bool:
    """
    Detecta si la respuesta de la IA es irrelevante, vacía o genérica.
    """
    patrones_no_entendi = [
        r"no entend[íi]",
        r"no comprendo",
        r"no puedo responder",
        r"no tengo información",
        r"no sé",
        r"no puedo ayudarte",
        r"no puedo contestar",
        r"no tengo datos",
        r"no tengo suficiente información",
        r"no tengo conocimiento",
        r"no puedo procesar",
        r"no puedo realizar esa acción",
        r"no puedo hacer eso",
        r"no puedo responder esa pregunta",
        r"no puedo responder esa consulta",
        r"no puedo responder esa solicitud",
        r"no puedo responder esa petición",
        r"no puedo responder esa orden",
        r"no puedo responder esa instrucción",
        r"no puedo responder esa indicación",
        r"no puedo responder esa sugerencia",
        r"no puedo responder esa recomendación",
        r"no puedo responder esa observación",
        r"no puedo responder esa aclaración",
        r"no puedo responder esa corrección",
        r"no puedo responder esa explicación"
    ]
    for patron in patrones_no_entendi:
        if re.search(patron, respuesta, re.IGNORECASE):
            return True
    return False

def reforzar_respuesta_ia(respuesta: str, contexto: str = "") -> str:
    """
    Si la respuesta es genérica o poco útil, la reemplaza por una sugerencia proactiva o informativa.
    """
    if es_respuesta_no_entendida(respuesta) or not respuesta.strip():
        return (
            "Soy un asistente avanzado. Puedes preguntarme sobre recordatorios, eventos, agenda, "
            "o pedirme que te ayude a organizar tus tareas. Por ejemplo: "
            "\"Recuérdame llamar a Juan mañana a las 10\" o \"Agéndame una reunión el viernes a las 15\"."
        )
    return respuesta

def generar_respuesta_ia(
    tokenizer: Any,
    model: Any,
    system_prompt: str,
    history: Optional[List[Any]],
    text: str,
    max_length: int = 100,
    top_k: int = 50,
    top_p: float = 0.95,
    temperature: float = 0.7,
    num_return_sequences: int = 1
) -> Tuple[str, Any]:
    """
    Genera una respuesta usando un modelo de lenguaje conversacional.
    Devuelve la respuesta y el nuevo chat_history_ids.
    """
    try:
        # Preparamos el texto de entrada con el historial
        input_text = system_prompt + "\n"
        
        if history:
            for msg in history[-5:]:  # Usar solo los últimos 5 mensajes para no exceder el tamaño máximo
                role = "Usuario" if msg.get("role") == "user" else "Asistente"
                input_text += f"{role}: {msg.get('content', '')}\n"
        
        input_text += f"Usuario: {text}\nAsistente:"
        
        # Codificamos la entrada
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        # Generamos la respuesta
        outputs = model.generate(
            inputs["input_ids"],
            max_length=len(inputs["input_ids"][0]) + max_length,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decodificamos la respuesta
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extraemos solo la parte de la respuesta del asistente
        response = response.split("Asistente:")[-1].strip()
        
        # Limpiamos la respuesta
        response = limpiar_respuesta(response)
        
        # Si la respuesta es muy genérica, intentamos mejorarla
        if es_respuesta_no_entendida(response):
            response = reforzar_respuesta_ia(response, "¿Podrías ser más específico con tu pregunta?")
        
        return response, None
        
    except Exception as e:
        logging.error(f"Error al generar respuesta: {str(e)}")
        return "Lo siento, ha ocurrido un error al procesar tu solicitud. Por favor, inténtalo de nuevo más tarde.", None
