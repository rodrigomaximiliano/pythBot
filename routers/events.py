from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from datetime import datetime

router = APIRouter(prefix="/events", tags=["events"])

class Event(BaseModel):
    id: int
    title: str
    description: str
    datetime: datetime

events_db: List[Event] = []

@router.post("/")
def create_event(event: Event):
    events_db.append(event)
    return {"message": "Evento creado", "event": event}

@router.get("/")
def list_events():
    return events_db
