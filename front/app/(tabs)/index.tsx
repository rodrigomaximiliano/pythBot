import React, { useState, useEffect, useRef } from 'react';
import { 
  StyleSheet, 
  TextInput, 
  View, 
  ScrollView, 
  ActivityIndicator, 
  KeyboardAvoidingView, 
  Platform, 
  TouchableOpacity 
} from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';

import { API_URL } from '@constants/Api';
import { ThemedText } from '@components/ThemedText';
import { ThemedView } from '@components/ThemedView';

// Definir tipos para los mensajes del chat
interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  chatContainer: {
    flex: 1,
    padding: 8,
  },
  messageList: {
    flex: 1,
  },
  messageContainer: {
    maxWidth: '80%',
    marginVertical: 4,
    padding: 12,
    borderRadius: 16,
  },
  userMessage: {
    alignSelf: 'flex-end',
    backgroundColor: '#007AFF',
    borderBottomRightRadius: 4,
  },
  botMessage: {
    alignSelf: 'flex-start',
    backgroundColor: '#e9e9eb',
    borderBottomLeftRadius: 4,
  },
  messageText: {
    fontSize: 16,
    lineHeight: 20,
  },
  userMessageText: {
    color: '#fff',
  },
  botMessageText: {
    color: '#000',
  },
  inputContainer: {
    flexDirection: 'row',
    padding: 12,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
  },
  textInput: {
    flex: 1,
    minHeight: 48,
    maxHeight: 120,
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 24,
    backgroundColor: '#f0f0f0',
    fontSize: 16,
    color: '#000',
  },
  sendButton: {
    marginLeft: 8,
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#007AFF',
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingContainer: {
    padding: 16,
    alignItems: 'center',
  },
  timestamp: {
    fontSize: 12,
    color: '#666',
    marginTop: 4,
  },
});

export default function ChatScreen() {
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const scrollViewRef = useRef<ScrollView>(null);

  // Mensaje de bienvenida inicial
  useEffect(() => {
    setMessages([
      {
        id: '1',
        text: '¡Hola! Soy tu asistente personal. ¿En qué puedo ayudarte hoy?',
        sender: 'bot',
        timestamp: new Date(),
      },
    ]);
  }, []);

  // Desplazar hacia abajo cuando se agregan nuevos mensajes
  useEffect(() => {
    if (scrollViewRef.current) {
      setTimeout(() => {
        scrollViewRef.current?.scrollToEnd({ animated: true });
      }, 100);
    }
  }, [messages]);

  const sendMessage = async () => {
    if (!message.trim() || isLoading) {
      console.log('Mensaje vacío o carga en curso');
      return;
    }

    // Agregar mensaje del usuario
    const userMessage: Message = {
      id: Date.now().toString(),
      text: message.trim(),
      sender: 'user',
      timestamp: new Date(),
    };

    console.log('Agregando mensaje del usuario:', userMessage);
    setMessages(prev => [...prev, userMessage]);
    setMessage('');
    setIsLoading(true);

    try {
      console.log('=== INICIO DE LA SOLICITUD ===');
      console.log('URL de la API:', API_URL);
      
      // Crear un objeto con los datos a enviar
      const data = new URLSearchParams();
      data.append('message', userMessage.text);
      data.append('session_id', 'user-session-123');
      
      console.log('Datos a enviar:', {
        method: 'POST',
        url: API_URL,
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: data.toString(),
      });
      
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: data.toString(),
      });

      console.log('Respuesta recibida - Estado:', response.status, response.statusText);
      
      const responseData = await response.json().catch(e => {
        console.error('Error al parsear la respuesta JSON:', e);
        return null;
      });
      
      console.log('Datos de la respuesta:', responseData);
      
      if (!response.ok) {
        const errorMessage = typeof responseData === 'object' && responseData !== null && 'detail' in responseData 
          ? String(responseData.detail) 
          : `Error HTTP ${response.status}: ${response.statusText}`;
        console.error('Error en la respuesta:', errorMessage);
        throw new Error(errorMessage);
      }
      
      // Agregar respuesta del bot
      const botMessage: Message = {
        id: Date.now().toString(),
        text: (typeof responseData === 'object' && responseData !== null && 'response' in responseData)
          ? String(responseData.response)
          : 'No pude procesar tu solicitud',
        sender: 'bot',
        timestamp: new Date(),
      };

      console.log('Mensaje del bot creado:', botMessage);
      setMessages(prev => [...prev, botMessage]);
      console.log('=== FIN DE LA SOLICITUD EXITOSA ===');
    } catch (error) {
      console.error('=== ERROR EN LA SOLICITUD ===');
      console.error('Tipo de error:', error instanceof Error ? error.name : typeof error);
      console.error('Mensaje de error:', error instanceof Error ? error.message : String(error));
      console.error('Stack trace:', error instanceof Error ? error.stack : 'No disponible');
      console.error('=== FIN DEL ERROR ===');
      console.error('Error al enviar mensaje:', error);
      
      const errorMessage: Message = {
        id: Date.now().toString(),
        text: 'Lo siento, hubo un error al procesar tu mensaje. Por favor, inténtalo de nuevo.',
        sender: 'bot',
        timestamp: new Date(),
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: any) => {
    if (e.nativeEvent.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  // Renderizar un mensaje individual
  const renderMessage = (msg: Message) => {
    const isUser = msg.sender === 'user';
    const messageStyle = [
      styles.messageContainer,
      isUser ? styles.userMessage : styles.botMessage,
    ];
    const textStyle = [
      styles.messageText,
      isUser ? styles.userMessageText : styles.botMessageText,
    ];

    return (
      <View
        key={msg.id}
        style={[
          messageStyle,
          { alignSelf: isUser ? 'flex-end' : 'flex-start' },
        ]}
      >
        <ThemedText style={textStyle}>{msg.text}</ThemedText>
        <ThemedText style={styles.timestamp}>
          {formatTime(msg.timestamp)}
        </ThemedText>
      </View>
    );
  };

  return (
    <KeyboardAvoidingView 
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}
    >
      <View style={styles.chatContainer}>
        <ScrollView
          ref={scrollViewRef}
          style={styles.messageList}
          contentContainerStyle={{ paddingBottom: 20 }}
        >
          {messages.map(renderMessage)}
          {isLoading && (
            <View style={[styles.messageContainer, styles.botMessage, { alignSelf: 'flex-start' }]}>
              <ThemedText style={styles.messageText}>Escribiendo...</ThemedText>
            </View>
          )}
        </ScrollView>

        <View style={styles.inputContainer}>
          <TextInput
            style={styles.textInput}
            placeholder="Escribe un mensaje..."
            placeholderTextColor="#666"
            value={message}
            onChangeText={setMessage}
            multiline
            onSubmitEditing={sendMessage}
            returnKeyType="send"
            blurOnSubmit={false}
            editable={!isLoading}
          />
          <TouchableOpacity 
            style={[
              styles.sendButton,
              { opacity: message.trim() ? 1 : 0.5 },
            ]}
            onPress={sendMessage}
            disabled={!message.trim() || isLoading}
          >
            <MaterialIcons name="send" size={24} color="white" />
          </TouchableOpacity>
        </View>
      </View>
    </KeyboardAvoidingView>
  );
}
