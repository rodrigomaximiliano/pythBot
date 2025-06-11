from typing import Dict, List, Tuple
from datetime import datetime

# Estructura: {session_id: [ (texto, fecha) ]}
_recordatorios: Dict[str, List[Tuple[str, datetime]]] = {}

def agregar_recordatorio(session_id: str, texto: str, fecha: datetime) -> bool:
    """
    Agrega un recordatorio solo si la fecha es futura.
    Devuelve True si se agregó, False si la fecha es pasada.
    """
    if fecha < datetime.now():
        return False
    if session_id not in _recordatorios:
        _recordatorios[session_id] = []
    _recordatorios[session_id].append((texto, fecha))
    return True

def obtener_recordatorios(session_id: str) -> List[Tuple[str, datetime]]:
    return _recordatorios.get(session_id, [])

def obtener_recordatorios_futuros(session_id: str) -> List[Tuple[str, datetime]]:
    ahora = datetime.now()
    return [r for r in _recordatorios.get(session_id, []) if r[1] >= ahora]

def total_recordatorios(session_id: str) -> int:
    return len(_recordatorios.get(session_id, []))
