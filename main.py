from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from routers import chat_new as chat, reminders, events
import logging
from typing import Any, Dict
import traceback

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Configuración de la aplicación
app = FastAPI(
    title="API de Asistente Personal",
    description="API para la aplicación móvil de asistente personal con recordatorios y eventos",
    version="1.0.0"
)

# Configurar manualmente la documentación
app.router.redirect_slashes = True
app.router.redirect_slashes = False

# Configurar las rutas de documentación
app.router.add_api_route(
    "/docs",
    lambda: None,
    include_in_schema=False,
    methods=["GET"]
)

app.router.add_api_route(
    "/api/docs",
    lambda: None,
    include_in_schema=False,
    methods=["GET"]
)

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, reemplazar con los dominios permitidos
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "X-Total-Count"]
)

# Manejo de errores global
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Error no manejado: {str(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Error interno del servidor"},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Error de validación: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body},
    )

# Incluir routers con prefijo /api
api_prefix = "/api"

# Incluir el router del chat con el prefijo /api
app.include_router(chat.router, prefix=api_prefix)  # /api/chat/
app.include_router(reminders.router, prefix=f"{api_prefix}/reminders")
app.include_router(events.router, prefix=f"{api_prefix}/events")

# Función para imprimir las rutas disponibles
@app.on_event("startup")
async def print_routes():
    print("\nRutas disponibles:")
    for route in app.routes:
        if hasattr(route, 'methods'):
            print(f"{route.path} - {route.methods}")

@app.get("/", include_in_schema=False)
async def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/api/docs")
