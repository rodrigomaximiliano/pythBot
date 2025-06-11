from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from datetime import datetime

router = APIRouter(prefix="/reminders", tags=["reminders"])

class Reminder(BaseModel):
    id: int
    text: str
    datetime: datetime

reminders_db: List[Reminder] = []

@router.post("/")
def create_reminder(reminder: Reminder):
    reminders_db.append(reminder)
    return {"message": "Recordatorio creado", "reminder": reminder}

@router.get("/")
def list_reminders():
    return reminders_db
