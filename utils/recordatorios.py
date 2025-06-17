"""
Módulo para el manejo de recordatorios.
Incluye funciones para crear, listar y gestionar recordatorios.
"""
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import uuid
import json # Importar json
import os # Importar os
from .date_utils import get_current_datetime, is_future_datetime, format_datetime, get_time_until

# Nombre del archivo para guardar los recordatorios
RECORDATORIOS_FILE = "data/recordatorios.json"

# Asegurarse de que el directorio de datos exista
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

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
        return f"{self.texto} ({tiempo_restante} - {format_datetime(self.fecha)})"

# Estructura: {session_id: {recordatorio_id: Recordatorio}}
_recordatorios: Dict[str, Dict[str, Recordatorio]] = {}

def _guardar_recordatorios():
    """Guarda los recordatorios en un archivo JSON."""
    data_to_save = {}
    for session_id, recordatorios_sesion in _recordatorios.items():
        data_to_save[session_id] = {}
        for recordatorio_id, recordatorio in recordatorios_sesion.items():
            data_to_save[session_id][recordatorio_id] = {
                "id": recordatorio.id,
                "texto": recordatorio.texto,
                "fecha": recordatorio.fecha.isoformat(), # Convertir datetime a string ISO
                "recurrente": recordatorio.recurrente,
                "intervalo": str(recordatorio.intervalo) if recordatorio.intervalo else None, # Convertir timedelta a string
                "max_repeticiones": recordatorio.max_repeticiones,
                "repeticiones": recordatorio.repeticiones,
                "activo": recordatorio.activo
            }
    
    with open(RECORDATORIOS_FILE, "w") as f:
        json.dump(data_to_save, f, indent=4)

def _cargar_recordatorios():
    """Carga los recordatorios desde un archivo JSON."""
    global _recordatorios
    if not os.path.exists(RECORDATORIOS_FILE):
        _recordatorios = {}
        return
        
    with open(RECORDATORIOS_FILE, "r") as f:
        try:
            data_loaded = json.load(f)
            _recordatorios = {}
            for session_id, recordatorios_sesion_data in data_loaded.items():
                _recordatorios[session_id] = {}
                for recordatorio_id, recordatorio_data in recordatorios_sesion_data.items():
                    # Convertir string ISO a datetime
                    fecha = datetime.fromisoformat(recordatorio_data["fecha"])
                    # Convertir string de intervalo a timedelta
                    intervalo = timedelta() # Valor por defecto
                    if recordatorio_data.get("intervalo"):
                         # Esto es una conversión básica, puede necesitar mejoras
                         try:
                             # Asumimos formato "DD days, HH:MM:SS" o similar de str(timedelta)
                             parts = recordatorio_data["intervalo"].split(',')
                             days = 0
                             seconds = 0
                             for part in parts:
                                 part = part.strip()
                                 if 'day' in part:
                                     days = int(part.split(' ')[0])
                                 elif ':' in part:
                                     h, m, s = map(int, part.split(':'))
                                     seconds = h * 3600 + m * 60 + s
                             intervalo = timedelta(days=days, seconds=seconds)
                         except Exception as e:
                             print(f"Error al parsear intervalo: {recordatorio_data.get('intervalo')} - {e}")
                             intervalo = None # Si falla el parseo, establecer a None


                    _recordatorios[session_id][recordatorio_id] = Recordatorio(
                        id=recordatorio_data["id"],
                        texto=recordatorio_data["texto"],
                        fecha=fecha,
                        recurrente=recordatorio_data.get("recurrente", False),
                        intervalo=intervalo,
                        max_repeticiones=recordatorio_data.get("max_repeticiones"),
                        repeticiones=recordatorio_data.get("repeticiones", 0),
                        activo=recordatorio_data.get("activo", True)
                    )
        except json.JSONDecodeError:
            # Si el archivo está vacío o corrupto, inicializar recordatorios vacíos
            _recordatorios = {}
        except Exception as e:
            print(f"Error al cargar recordatorios: {e}")
            _recordatorios = {} # En caso de otros errores de carga

# Cargar recordatorios al iniciar el módulo
_cargar_recordatorios()

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
    _guardar_recordatorios() # Guardar después de agregar
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
        _guardar_recordatorios() # Guardar después de eliminar
        return True
    
    # Manejar recordatorio recurrente
    if (recordatorio.max_repeticiones is not None and
        recordatorio.repeticiones >= recordatorio.max_repeticiones):
        # Se alcanzó el máximo de repeticiones
        del _recordatorios[session_id][recordatorio_id]
        _guardar_recordatorios() # Guardar después de eliminar
        return True
    
    # Programar próxima repetición
    recordatorio.repeticiones += 1
    recordatorio.fecha += recordatorio.intervalo
    
    # Si la nueva fecha ya pasó, intentar con la siguiente repetición
    if not is_future_datetime(recordatorio.fecha):
        # Llamada recursiva para encontrar la próxima fecha válida y guardar
        result = marcar_como_completado(session_id, recordatorio_id)
        if result: # Si la llamada recursiva tuvo éxito (encontró una fecha futura o eliminó)
             _guardar_recordatorios() # Guardar después de la actualización recursiva
        return result
    
    _guardar_recordatorios() # Guardar después de actualizar la fecha
    return True

def eliminar_recordatorio(session_id: str, recordatorio_id: str) -> bool:
    """Elimina un recordatorio."""
    if (session_id in _recordatorios and
        recordatorio_id in _recordatorios[session_id]):
        del _recordatorios[session_id][recordatorio_id]
        _guardar_recordatorios() # Guardar después de eliminar
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
    _guardar_recordatorios() # Guardar después de limpiar
    
    return total
