"""
Módulo para el manejo de recordatorios.
Incluye funciones para crear, listar y gestionar recordatorios.
"""
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import uuid
from .date_utils import get_current_datetime, is_future_datetime, format_datetime, get_time_until

class Recordatorio:
    """Clase que representa un recordatorio."""
    
    def __init__(self, id: str, texto: str, fecha: datetime, recurrente: bool = False, 
                 intervalo: Optional[timedelta] = None, max_repeticiones: Optional[int] = None):
        self.id = id
        self.texto = texto
        self.fecha = fecha
        self.recurrente = recurrente
        self.intervalo = intervalo
        self.max_repeticiones = max_repeticiones
        self.repeticiones = 0
        self.activo = True
    
    def __str__(self) -> str:
        tiempo_restante = get_time_until(self.fecha)
        return f"{self.texto} ({tiempo_remaining} - {format_datetime(self.fecha)})"

# Estructura: {session_id: {recordatorio_id: Recordatorio}}
_recordatorios: Dict[str, Dict[str, Recordatorio]] = {}

def agregar_recordatorio(
    session_id: str, 
    texto: str, 
    fecha: datetime,
    recurrente: bool = False,
    intervalo: Optional[timedelta] = None,
    max_repeticiones: Optional[int] = None
) -> Tuple[bool, str]:
    """
    Agrega un nuevo recordatorio.
    
    Args:
        session_id: ID de la sesión del usuario
        texto: Texto del recordatorio
        fecha: Fecha y hora del recordatorio
        recurrente: Si es un recordatorio recurrente
        intervalo: Intervalo de repetición (para recordatorios recurrentes)
        max_repeticiones: Número máximo de repeticiones (opcional)
        
    Returns:
        Tuple[bool, str]: (éxito, mensaje)
    """
    if not texto or not fecha:
        return False, "Faltan datos del recordatorio"
        
    # Validar fecha futura
    if not is_future_datetime(fecha):
        return False, "No se pueden crear recordatorios en el pasado"
    
    # Validar configuración de recordatorio recurrente
    if recurrente and not intervalo:
        return False, "Se debe especificar un intervalo para recordatorios recurrentes"
    
    # Inicializar diccionario de recordatorios para la sesión si no existe
    if session_id not in _recordatorios:
        _recordatorios[session_id] = {}
    
    # Crear ID único para el recordatorio
    recordatorio_id = str(uuid.uuid4())
    
    # Crear y guardar el recordatorio
    recordatorio = Recordatorio(
        id=recordatorio_id,
        texto=texto,
        fecha=fecha,
        recurrente=recurrente,
        intervalo=intervalo,
        max_repeticiones=max_repeticiones
    )
    
    _recordatorios[session_id][recordatorio_id] = recordatorio
    return True, f"Recordatorio creado para {format_datetime(fecha)}"

def obtener_recordatorios(session_id: str, solo_activos: bool = True) -> List[Recordatorio]:
    """
    Obtiene todos los recordatorios de una sesión.
    
    Args:
        session_id: ID de la sesión
        solo_activos: Si es True, solo devuelve recordatorios activos
        
    Returns:
        List[Recordatorio]: Lista de recordatorios
    """
    if session_id not in _recordatorios:
        return []
        
    recordatorios = list(_recordatorios[session_id].values())
    
    if solo_activos:
        recordatorios = [r for r in recordatorios if r.activo]
    
    # Ordenar por fecha más cercana
    return sorted(recordatorios, key=lambda x: x.fecha)

def obtener_recordatorios_proximos(session_id: str, horas: int = 24) -> List[Recordatorio]:
    """
    Obtiene los recordatorios que vencen en las próximas horas.
    
    Args:
        session_id: ID de la sesión
        horas: Rango de horas a buscar
        
    Returns:
        List[Recordatorio]: Lista de recordatorios próximos
    """
    ahora = get_current_datetime()
    limite = ahora + timedelta(hours=horas)
    
    recordatorios = obtener_recordatorios(session_id)
    return [r for r in recordatorios if ahora <= r.fecha <= limite]

def marcar_como_completado(session_id: str, recordatorio_id: str) -> bool:
    """
    Marca un recordatorio como completado.
    Para recordatorios recurrentes, programa la próxima repetición.
    """
    if (session_id not in _recordatorios or 
        recordatorio_id not in _recordatorios[session_id]):
        return False
    
    recordatorio = _recordatorios[session_id][recordatorio_id]
    
    if not recordatorio.recurrente:
        # Eliminar recordatorio no recurrente
        del _recordatorios[session_id][recordatorio_id]
        return True
    
    # Manejar recordatorio recurrente
    if (recordatorio.max_repeticiones is not None and 
        recordatorio.repeticiones >= recordatorio.max_repeticiones):
        # Se alcanzó el máximo de repeticiones
        del _recordatorios[session_id][recordatorio_id]
        return True
    
    # Programar próxima repetición
    recordatorio.repeticiones += 1
    recordatorio.fecha += recordatorio.intervalo
    
    # Si la nueva fecha ya pasó, intentar con la siguiente repetición
    if not is_future_datetime(recordatorio.fecha):
        return marcar_como_completado(session_id, recordatorio_id)
    
    return True

def eliminar_recordatorio(session_id: str, recordatorio_id: str) -> bool:
    """Elimina un recordatorio."""
    if (session_id in _recordatorios and 
        recordatorio_id in _recordatorios[session_id]):
        del _recordatorios[session_id][recordatorio_id]
        return True
    return False

def total_recordatorios(session_id: str, solo_activos: bool = True) -> int:
    """
    Devuelve el número total de recordatorios para una sesión.
    """
    if session_id not in _recordatorios:
        return 0
    
    if solo_activos:
        return sum(1 for r in _recordatorios[session_id].values() if r.activo)
    
    return len(_recordatorios[session_id])

def limpiar_recordatorios_completados(session_id: str) -> int:
    """
    Elimina todos los recordatorios completados.
    Devuelve el número de recordatorios eliminados.
    """
    if session_id not in _recordatorios:
        return 0
    
    # Los recordatorios completados se eliminan automáticamente,
    # así que esta función solo limpia los inactivos
    total = 0
    for recordatorio_id in list(_recordatorios[session_id].keys()):
        if not _recordatorios[session_id][recordatorio_id].activo:
            del _recordatorios[session_id][recordatorio_id]
            total += 1
    
    return total
