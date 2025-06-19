# Asistente Personal con NLP Avanzado

Un asistente virtual inteligente que utiliza procesamiento de lenguaje natural (NLP) avanzado para entender y responder a solicitudes en español. El sistema puede manejar recordatorios, responder preguntas y mantener conversaciones naturales.

## Características Principales

- 🎯 **Comprensión de lenguaje natural** usando BERT multilingüe
- 📅 **Gestión de recordatorios** con fechas y horas
- 💬 **Conversación natural** con múltiples variaciones de respuestas
- 🧠 **Aprendizaje por ejemplos** para mejorar la precisión
- 🚀 **Arquitectura escalable** para añadir nuevas funcionalidades


![Captura de pantalla 2025-06-19 173703](https://github.com/user-attachments/assets/4ac0a1ef-b305-4938-ad3f-d79975dc641f)


## Estructura del Proyecto

```
pythBot/
├── nlp/
│   └── intent_classifier.py  # Motor de NLP y clasificación de intenciones
├── routers/
│   └── chat_new.py          # Rutas de la API y lógica principal
├── main.py                   # Punto de entrada de la aplicación
└── README.md                 # Este archivo
```

## Cómo Funciona

### 1. Procesamiento de Mensajes

1. **Recepción del mensaje**: El usuario envía un mensaje a través de la API.
2. **Clasificación de intención**: El motor de NLP analiza el mensaje para determinar la intención del usuario.
3. **Extracción de entidades**: Se identifican elementos clave como fechas, horas y tareas.
4. **Generación de respuesta**: Se genera una respuesta natural basada en la intención y entidades detectadas.

### 2. Motor de NLP

El sistema utiliza el modelo `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` para:

- Convertir texto en vectores numéricos (embeddings)
- Calcular similitud semántica entre frases
- Clasificar intenciones basado en ejemplos de entrenamiento

### 3. Tipos de Intenciones

- **Saludos**: "Hola", "Buenos días", "¿Cómo estás?"
- **Despedidas**: "Adiós", "Hasta luego"
- **Agradecimientos**: "Gracias", "Te lo agradezco"
- **Ayuda**: "¿Qué puedes hacer?", "¿Cómo funcionas?"
- **Crear recordatorio**: "Recuérdame comprar leche mañana"
- **Listar recordatorios**: "¿Qué recordatorios tengo?"

## Uso

### Requisitos

- Python 3.8+
- Bibliotecas listadas en `requirements.txt`

### Instalación

1. Clona el repositorio:
   ```bash
   git clone [URL_DEL_REPOSITORIO]
   cd pythBot
   ```

2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Ejecuta la aplicación:
   ```bash
   uvicorn main:app --reload
   ```

### Uso de la API

```http
POST /api/chat/
Content-Type: application/x-www-form-urlencoded

message=Recuérdame comprar leche mañana&session_id=usuario123
```

**Respuesta:**
```json
{
  "response": "✅ He creado un recordatorio para: comprar leche mañana",
  "status": "success",
  "intent": "create_reminder",
  "entities": {
    "task": "comprar leche",
    "date": [{"value": "mañana"}]
  }
}
```

## Personalización

### Agregar más ejemplos

Puedes mejorar la precisión añadiendo más ejemplos en `intent_classifier.py` dentro del diccionario `self.examples`.

### Añadir nuevas intenciones

1. Agrega un nuevo tipo en el enum `Intent`
2. Añade ejemplos en `self.examples`
3. Implementa el manejo en `generate_response`

## Mejoras Futuras

- [ ] Integrar con un calendario (Google Calendar, Outlook)
- [ ] Añadir reconocimiento de voz
- [ ] Implementar un sistema de aprendizaje continuo
- [ ] Crear una interfaz web amigable

## Contribución

¡Las contribuciones son bienvenidas! Por favor, lee las guías de contribución antes de enviar un pull request.

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.
