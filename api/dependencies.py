"""
Dependencias comunes para los endpoints de la API.
"""
from fastapi import Depends, HTTPException, status
from typing import Generator, Optional

# Dependencia para obtener el ID de sesión
def get_session_id(session_id: Optional[str] = None) -> str:
    """Obtiene o genera un ID de sesión para el usuario."""
    return session_id or "default_session"

# Dependencia para verificar autenticación (puedes personalizarla según tus necesidades)
def get_current_user(token: Optional[str] = None):
    """Verifica si el usuario está autenticado."""
    # TODO: Implementar lógica de autenticación real
    if not token:
        from api.errors import UnauthorizedError
        raise UnauthorizedError("Se requiere autenticación")
    return {"username": "usuario_ejemplo"}  # Usuario de ejemplo

# Dependencia para manejo de paginación
def get_pagination_params(
    skip: int = 0,
    limit: int = 10,
    max_limit: int = 100
):
    """Valida y devuelve los parámetros de paginación."""
    if skip < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El parámetro 'skip' no puede ser negativo"
        )
    if limit <= 0 or limit > max_limit:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"El parámetro 'limit' debe estar entre 1 y {max_limit}"
        )
    return {"skip": skip, "limit": limit}
