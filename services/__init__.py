"""
Módulo de servicios de la aplicación.

Este módulo exporta las instancias de los servicios para ser utilizados en toda la aplicación.
"""
from .chat_service import chat_service

# Exportar la instancia del servicio de chat
__all__ = ['chat_service']
