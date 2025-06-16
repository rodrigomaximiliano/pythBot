""
Módulo principal para la configuración central de la aplicación.
"""
from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Configuración de la aplicación."""
    APP_NAME: str = "Asistente Personal"
    DEBUG: bool = True
    API_V1_STR: str = "/api/v1"
    
    # Configuración de la base de datos (si es necesaria)
    DATABASE_URL: Optional[str] = None
    
    # Configuración de autenticación
    SECRET_KEY: str = "clave-secreta-temporal-cambiar-en-produccion"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 días
    
    class Config:
        case_sensitive = True
        env_file = ".env"

# Instancia de configuración
settings = Settings()
