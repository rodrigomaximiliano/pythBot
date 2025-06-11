from fastapi import FastAPI
from routers import chat, reminders, events
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

app = FastAPI()

app.include_router(chat.router)
app.include_router(reminders.router)
app.include_router(events.router)

@app.get("/")
def read_root():
    return {"message": "Backend funcionando correctamente"}
