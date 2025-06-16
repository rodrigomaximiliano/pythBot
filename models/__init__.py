"""
Modelos de datos de la aplicación.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

# Modelos base
class MessageBase(BaseModel):
    """Modelo base para mensajes."""
    content: str
    role: str = "user"  # 'user' o 'assistant'
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None

class ChatMessage(MessageBase):
    """Modelo para mensajes de chat."""
    session_id: str

class ReminderCreate(BaseModel):
    """Modelo para crear recordatorios."""
    text: str
    due_date: datetime
    metadata: Optional[Dict[str, Any]] = None

class Reminder(ReminderCreate):
    """Modelo para recordatorios."""
    id: str
    created_at: datetime
    is_completed: bool = False

class EventCreate(BaseModel):
    """Modelo para crear eventos."""
    title: str
    description: str
    start_time: datetime
    end_time: datetime
    location: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class Event(EventCreate):
    """Modelo para eventos."""
    id: str
    created_at: datetime
    updated_at: datetime
