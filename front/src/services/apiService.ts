import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse, AxiosError } from 'axios';

// Configuración de la API
const API_CONFIG = {
  // URL base de la API (ajusta el puerto según tu configuración)
  BASE_URL: 'http://10.0.2.2:8000/api',  // Para Android Emulator
  // BASE_URL: 'http://localhost:8000/api',  // Para iOS o Android con reverse proxy
  // BASE_URL: 'http://TU_IP_LOCAL:8000/api',  // Para dispositivos físicos (reemplaza TU_IP_LOCAL)
  
  // Endpoints
  ENDPOINTS: {
    CHAT: '/chat',
    REMINDERS: '/reminders',
    EVENTS: '/events'
  },
  
  // Configuración de tiempo de espera para las peticiones (en milisegundos)
  TIMEOUT: 10000,
  
  // Configuración de reintentos
  MAX_RETRIES: 3,
  RETRY_DELAY: 1000
};

// Función para obtener la URL completa de un endpoint
const getApiUrl = (endpoint: string): string => {
  return `${API_CONFIG.BASE_URL}${endpoint}`;
};

// Configuración de axios
const apiClient: AxiosInstance = axios.create({
  baseURL: API_CONFIG.BASE_URL,
  timeout: API_CONFIG.TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
});

// Interceptor para añadir token de autenticación si existe
apiClient.interceptors.request.use(
  (config) => {
    // Aquí puedes añadir lógica para incluir el token de autenticación
    // const token = await AsyncStorage.getItem('userToken');
    // if (token) {
    //   config.headers.Authorization = `Bearer ${token}`;
    // }
    return config;
  },
  (error: AxiosError) => {
    return Promise.reject(error);
  }
);

// Interceptor para manejar errores globales
apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    return response;
  },
  async (error: AxiosError) => {
    // Manejo de errores global
    if (error.response) {
      // El servidor respondió con un código de estado fuera del rango 2xx
      console.error('Error en la respuesta:', error.response.status, error.response.data);
    } else if (error.request) {
      // La solicitud fue hecha pero no se recibió respuesta
      console.error('No se recibió respuesta del servidor');
    } else {
      // Algo sucedió al configurar la solicitud
      console.error('Error al configurar la solicitud:', error.message);
    }
    return Promise.reject(error);
  }
);

// Servicio de chat
export const chatService = {
  sendMessage: async (message: string, sessionId: string = 'default'): Promise<any> => {
    try {
      const response = await apiClient.post(getApiUrl(API_CONFIG.ENDPOINTS.CHAT), {
        message,
        session_id: sessionId,
      });
      return response.data;
    } catch (error) {
      console.error('Error al enviar mensaje:', error);
      throw error;
    }
  },
  
  sendAudio: async (audioUri: string, sessionId: string = 'default'): Promise<any> => {
    try {
      const formData = new FormData();
      formData.append('audio', {
        uri: audioUri,
        type: 'audio/m4a',
        name: 'audio.m4a',
      } as any);
      formData.append('session_id', sessionId);
      
      const response = await apiClient.post(getApiUrl(API_CONFIG.ENDPOINTS.CHAT), formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      console.error('Error al enviar audio:', error);
      throw error;
    }
  }
};

// Servicio de recordatorios
export const reminderService = {
  getReminders: async (sessionId: string = 'default'): Promise<any> => {
    try {
      const response = await apiClient.get(getApiUrl(API_CONFIG.ENDPOINTS.REMINDERS), {
        params: { session_id: sessionId }
      });
      return response.data;
    } catch (error) {
      console.error('Error al obtener recordatorios:', error);
      throw error;
    }
  },
  
  createReminder: async (text: string, datetime: string, sessionId: string = 'default'): Promise<any> => {
    try {
      const response = await apiClient.post(getApiUrl(API_CONFIG.ENDPOINTS.REMINDERS), {
        text,
        datetime,
        session_id: sessionId,
      });
      return response.data;
    } catch (error) {
      console.error('Error al crear recordatorio:', error);
      throw error;
    }
  },
  
  deleteReminder: async (reminderId: number, sessionId: string = 'default'): Promise<void> => {
    try {
      await apiClient.delete(`${getApiUrl(API_CONFIG.ENDPOINTS.REMINDERS)}/${reminderId}`, {
        params: { session_id: sessionId }
      });
    } catch (error) {
      console.error('Error al eliminar recordatorio:', error);
      throw error;
    }
  }
};

// Servicio de eventos
export const eventService = {
  getEvents: async (sessionId: string = 'default', startDate?: string, endDate?: string): Promise<any> => {
    try {
      const params: any = { session_id: sessionId };
      if (startDate) params.start_date = startDate;
      if (endDate) params.end_date = endDate;
      
      const response = await apiClient.get(getApiUrl(API_CONFIG.ENDPOINTS.EVENTS), { params });
      return response.data;
    } catch (error) {
      console.error('Error al obtener eventos:', error);
      throw error;
    }
  },
  
  createEvent: async (title: string, description: string, datetime: string, sessionId: string = 'default'): Promise<any> => {
    try {
      const response = await apiClient.post(getApiUrl(API_CONFIG.ENDPOINTS.EVENTS), {
        title,
        description,
        datetime,
        session_id: sessionId,
      });
      return response.data;
    } catch (error) {
      console.error('Error al crear evento:', error);
      throw error;
    }
  },
  
  deleteEvent: async (eventId: number, sessionId: string = 'default'): Promise<void> => {
    try {
      await apiClient.delete(`${getApiUrl(API_CONFIG.ENDPOINTS.EVENTS)}/${eventId}`, {
        params: { session_id: sessionId }
      });
    } catch (error) {
      console.error('Error al eliminar evento:', error);
      throw error;
    }
  }
};

export default apiClient;
