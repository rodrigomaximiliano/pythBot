"""
Módulo para el manejo de eventos.
Incluye funciones para crear, listar y gestionar eventos.
"""
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import uuid
import json # Importar json
import os # Importar os
# Asumiendo que date_utils también es útil para eventos
from .date_utils import get_current_datetime, is_future_datetime, format_datetime, get_time_until

# Nombre del archivo para guardar los eventos
EVENTOS_FILE = "data/eventos.json"

# Asegurarse de que el directorio de datos exista (ya lo hicimos en recordatorios, pero lo repetimos por seguridad)
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

class Evento:
    """Clase que representa un evento."""
    
    def __init__(self, id: str, titulo: str, fecha: datetime, ubicacion: Optional[str] = None, descripcion: Optional[str] = None):
        self.id = id
        self.titulo = titulo
        self.fecha = fecha
        self.ubicacion = ubicacion
        self.descripcion = descripcion
        self.activo = True # Podríamos usar esto para eventos pasados o cancelados
    
    def __str__(self) -> str:
        # Formato básico, se puede mejorar
        details = f"{self.titulo} ({format_datetime(self.fecha)})"
        if self.ubicacion:
            details += f" en {self.ubicacion}"
        if self.descripcion:
            details += f": {self.descripcion}"
        return details

# Estructura: {session_id: {evento_id: Evento}}
_eventos: Dict[str, Dict[str, Evento]] = {}

def _guardar_eventos():
    """Guarda los eventos en un archivo JSON."""
    data_to_save = {}
    for session_id, eventos_sesion in _eventos.items():
        data_to_save[session_id] = {}
        for evento_id, evento in eventos_sesion.items():
            data_to_save[session_id][evento_id] = {
                "id": evento.id,
                "titulo": evento.titulo,
                "fecha": evento.fecha.isoformat(), # Convertir datetime a string ISO
                "ubicacion": evento.ubicacion,
                "descripcion": evento.descripcion,
                "activo": evento.activo
            }
    
    with open(EVENTOS_FILE, "w") as f:
        json.dump(data_to_save, f, indent=4)

def _cargar_eventos():
    """Carga los eventos desde un archivo JSON."""
    global _eventos
    if not os.path.exists(EVENTOS_FILE):
        _eventos = {}
        return
        
    with open(EVENTOS_FILE, "r") as f:
        try:
            data_loaded = json.load(f)
            _eventos = {}
            for session_id, eventos_sesion_data in data_loaded.items():
                _eventos[session_id] = {}
                for evento_id, evento_data in eventos_sesion_data.items():
                    # Convertir string ISO a datetime
                    fecha = datetime.fromisoformat(evento_data["fecha"])

                    _eventos[session_id][evento_id] = Evento(
                        id=evento_data["id"],
                        titulo=evento_data["titulo"],
                        fecha=fecha,
                        ubicacion=evento_data.get("ubicacion"),
                        descripcion=evento_data.get("descripcion"),
                        activo=evento_data.get("activo", True)
                    )
        except json.JSONDecodeError:
            # Si el archivo está vacío o corrupto, inicializar eventos vacíos
            _eventos = {}
        except Exception as e:
            print(f"Error al cargar eventos: {e}")
            _eventos = {} # En caso de otros errores de carga

# Cargar eventos al iniciar el módulo
_cargar_eventos()

def agregar_evento(
    session_id: str, 
    titulo: str, 
    fecha: datetime,
    ubicacion: Optional[str] = None,
    descripcion: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Agrega un nuevo evento.
    
    Args:
        session_id: ID de la sesión del usuario
        titulo: Título del evento
        fecha: Fecha y hora del evento
        ubicacion: Ubicación del evento (opcional)
        descripcion: Descripción del evento (opcional)
        
    Returns:
        Tuple[bool, str]: (éxito, mensaje)
    """
    if not titulo or not fecha:
        return False, "Faltan datos del evento (título o fecha)"
        
    # Validar fecha futura (opcional, podrías permitir eventos pasados)
    if not is_future_datetime(fecha):
         # Podríamos ajustar este mensaje o permitir eventos pasados
        return False, "No se pueden crear eventos en el pasado (por ahora)"
    
    # Inicializar diccionario de eventos para la sesión si no existe
    if session_id not in _eventos:
        _eventos[session_id] = {}
    
    # Crear ID único para el evento
    evento_id = str(uuid.uuid4())
    
    # Crear y guardar el evento
    evento = Evento(
        id=evento_id,
        titulo=titulo,
        fecha=fecha,
        ubicacion=ubicacion,
        descripcion=descripcion
    )
    
    _eventos[session_id][evento_id] = evento
    _guardar_eventos() # Guardar después de agregar
    return True, f"Evento '{titulo}' creado para {format_datetime(fecha)}"

def obtener_eventos(session_id: str, solo_activos: bool = True) -> List[Evento]:
    """
    Obtiene todos los eventos de una sesión.
    
    Args:
        session_id: ID de la sesión
        solo_activos: Si es True, solo devuelve eventos activos
        
    Returns:
        List[Evento]: Lista de eventos
    """
    if session_id not in _eventos:
        return []
        
    eventos = list(_eventos[session_id].values())
    
    if solo_activos:
        # Podríamos definir la lógica de "activo" basada en la fecha actual
        eventos = [e for e in eventos if e.activo and is_future_datetime(e.fecha)]
    
    # Ordenar por fecha más cercana
    return sorted(eventos, key=lambda x: x.fecha)

def eliminar_evento(session_id: str, evento_id: str) -> bool:
    """Elimina un evento."""
    if (session_id in _eventos and
        evento_id in _eventos[session_id]):
        del _eventos[session_id][evento_id]
        _guardar_eventos() # Guardar después de eliminar
        return True
    return False

def total_eventos(session_id: str, solo_activos: bool = True) -> int:
    """
    Devuelve el número total de eventos para una sesión.
    """
    if session_id not in _eventos:
        return 0
    
    if solo_activos:
         return sum(1 for e in _eventos[session_id].values() if e.activo and is_future_datetime(e.fecha))
    
    return len(_eventos[session_id])

# Podríamos añadir más funciones como obtener_eventos_proximos, etc.