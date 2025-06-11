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
    max_length: int = 1000,
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
        system_input_ids = tokenizer.encode(system_prompt + tokenizer.eos_token, return_tensors="pt")
        limited_history = history[-3:] if history else []
        if limited_history:
            bot_input_ids = torch.cat(
                [system_input_ids] + limited_history + [tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")],
                dim=-1
            )
        else:
            bot_input_ids = torch.cat(
                [system_input_ids, tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")],
                dim=-1
            )

        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            num_return_sequences=num_return_sequences
        )
        respuesta = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        respuesta = limpiar_respuesta(respuesta)
        if not respuesta:
            respuesta = "No entendí tu mensaje, ¿puedes reformularlo?"
        return respuesta, chat_history_ids
    except Exception as e:
        logging.error(f"Error en generar_respuesta_ia: {e}")
        return "Ocurrió un error al generar la respuesta de IA.", None
