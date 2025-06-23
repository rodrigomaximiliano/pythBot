// Configuración básica para desarrollo local
const API_BASE_URL = 'http://localhost:8000';
const API_PREFIX = '/api';

// URL base de la API (sin la barra final)
export const API_BASE = `${API_BASE_URL}${API_PREFIX}`;

// URL completa para el endpoint de chat (con barra final para coincidir con el backend)
export const API_URL = `${API_BASE}/chat/`;

// Configuración para diferentes entornos
export const API_CONFIG = {
  API_URL: API_BASE,  // Usar API_BASE que ya incluye el prefijo
  API_PREFIX,
  TIMEOUT: 10000,  // 10 segundos de timeout
  
  // Configuraciones para diferentes entornos
  ENDPOINTS: {
    CHAT: `${API_BASE}/chat/`,
    REMINDERS: `${API_BASE}/reminders`,
    EVENTS: `${API_BASE}/events`
  }
};