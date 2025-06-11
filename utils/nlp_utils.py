import spacy
import dateparser
from datetime import datetime

# Carga el modelo de spaCy en español (solo una vez)
try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "es_core_news_sm"])
    nlp = spacy.load("es_core_news_sm")

def extraer_info_recordatorio(texto):
    doc = nlp(texto)
    fechas = [ent.text for ent in doc.ents if ent.label_ in ("DATE", "TIME")]
    texto_limpio = texto
    for ent in fechas:
        texto_limpio = texto_limpio.replace(ent, "")
    texto_limpio = texto_limpio.replace("recordar", "").replace("recordatorio", "").strip(":,.- ")
    fecha = None
    now = datetime.now()
    for ent in fechas:
        posible_fecha = dateparser.parse(ent, languages=['es'], settings={'RELATIVE_BASE': now})
        if posible_fecha:
            fecha = posible_fecha
            break
    return texto_limpio.strip(), fecha

def extraer_info_evento(texto):
    doc = nlp(texto)
    fechas = [ent.text for ent in doc.ents if ent.label_ in ("DATE", "TIME")]
    texto_limpio = texto
    for ent in fechas:
        texto_limpio = texto_limpio.replace(ent, "")
    texto_limpio = texto_limpio.replace("evento", "").replace("calendario", "").strip(":,.- ")
    fecha = None
    now = datetime.now()
    for ent in fechas:
        posible_fecha = dateparser.parse(ent, languages=['es'], settings={'RELATIVE_BASE': now})
        if posible_fecha:
            fecha = posible_fecha
            break
    partes = texto_limpio.split(".", 1)
    if len(partes) == 2:
        titulo = partes[0].strip()
        descripcion = partes[1].strip()
    else:
        partes = texto_limpio.split(",", 1)
        if len(partes) == 2:
            titulo = partes[0].strip()
            descripcion = partes[1].strip()
        else:
            titulo = texto_limpio.strip()
            descripcion = ""
    return titulo, descripcion, fecha

def detectar_intencion(texto):
    doc = nlp(texto)
    tiene_fecha = any(ent.label_ in ("DATE", "TIME") for ent in doc.ents)
    palabras_recordatorio = ["recordar", "recordatorio", "acuérdate", "avísame", "recuérdame", "anota", "apunta"]
    palabras_evento = ["evento", "calendario", "agenda", "cita", "reunión", "cumpleaños", "turno", "fiesta"]
    if any(p in texto for p in palabras_recordatorio) or (tiene_fecha and "recordar" in texto):
        return "recordatorio"
    if any(p in texto for p in palabras_evento) or (tiene_fecha and "evento" in texto):
        return "evento"
    if tiene_fecha and any(texto.startswith(v) for v in ["avísame", "acuérdate", "recuérdame", "anota", "apunta"]):
        return "recordatorio"
    if tiene_fecha and any(w in texto for w in ["tengo", "hay", "es", "será"]):
        return "evento"
    return None
