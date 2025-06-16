"""
Manejo centralizado de errores para la API.
"""
from fastapi import HTTPException, status
from typing import Any, Dict, Optional

class APIError(HTTPException):
    """Clase base para errores de la API."""
    def __init__(
        self,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail: Any = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=detail, headers=headers)

class NotFoundError(APIError):
    """Error 404: Recurso no encontrado."""
    def __init__(self, detail: str = "Recurso no encontrado") -> None:
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=detail)

class UnauthorizedError(APIError):
    """Error 401: No autorizado."""
    def __init__(self, detail: str = "No autorizado") -> None:
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )

class ValidationError(APIError):
    """Error 422: Error de validación."""
    def __init__(self, detail: Any = "Error de validación") -> None:
        super().__init__(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail)
