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
import { Colors } from '@constants/Colors';

import { API_URL } from '@constants/Api';
import { ThemedText } from '@components/ThemedText';
import { ThemedView } from '@components/ThemedView';

// Definir tipos para los mensajes del chat
interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  suggestions?: string[]; // Añadir campo opcional para sugerencias
  intent?: string; // Añadir campo opcional para la intención
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.light.background, // Usar color de fondo del tema
  },
  chatContainer: {
    flex: 1,
    paddingHorizontal: 10, // Espaciado horizontal
    paddingVertical: 5, // Espaciado vertical
  },
  messageList: {
    flex: 1,
  },
  messageContainer: {
    maxWidth: '85%', // Aumentar un poco el ancho máximo
    marginVertical: 5, // Espacio vertical entre mensajes
    padding: 10, // Relleno dentro de la burbuja
    borderRadius: 15, // Bordes más redondeados
    elevation: 1, // Sombra sutil para Android
    boxShadow: '0px 1px 1.5px rgba(0, 0, 0, 0.1)', // Sombra para iOS y web
  },
  userMessage: {
    alignSelf: 'flex-end',
    backgroundColor: Colors.light.tint, // Usar color de tinte del tema
    borderBottomRightRadius: 5, // Ajustar radio de la esquina inferior derecha
    boxShadow: '0px 1px 1.5px rgba(0, 0, 0, 0.1)'
  },
  botMessage: {
    alignSelf: 'flex-start',
    backgroundColor: '#E5E5EA', // Un gris claro para mensajes del bot
    boxShadow: '0px 1px 1.5px rgba(0, 0, 0, 0.1)',
    borderBottomLeftRadius: 5 // Ajustar radio de la esquina inferior izquierda
  },
  messageText: {
    fontSize: 16,
    lineHeight: 22, // Aumentar interlineado
    paddingHorizontal: 5, // Espaciado horizontal dentro del texto
  },
  userMessageText: {
    color: '#fff', // Texto blanco para mensajes del usuario (contrasta bien con el tinte)
  },
  botMessageText: {
    color: Colors.light.text, // Usar color de texto del tema
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center', // Centrar verticalmente los elementos
    paddingHorizontal: 10, // Espaciado horizontal
    paddingVertical: 8, // Espaciado vertical
    backgroundColor: Colors.light.background, // Usar color de fondo del tema
    borderTopWidth: 1,
    borderTopColor: '#ccc', // Borde más claro
  },
  textInput: {
    flex: 1,
    minHeight: 40, // Altura mínima
    maxHeight: 100, // Altura máxima
    paddingHorizontal: 15, // Espaciado horizontal
    paddingVertical: 10, // Espaciado vertical
    borderRadius: 20, // Bordes redondeados
    backgroundColor: '#f0f0f0', // Fondo ligeramente gris para el input
    fontSize: 16,
    color: Colors.light.text, // Usar color de texto del tema
    borderWidth: 0, // Eliminar borde
    // borderColor: '#e0e0e0', // Color del borde
    marginRight: 8, // Espacio a la derecha del input
  },
  sendButton: {
    width: 44, // Ancho del botón
    height: 44, // Altura del botón
    borderRadius: 22, // Bordes completamente redondeados
    backgroundColor: Colors.light.tint, // Usar color de tinte del tema
    justifyContent: 'center',
    alignItems: 'center',
    // Eliminar marginLeft ya que el TextInput tiene marginRight
  },
  loadingContainer: {
    padding: 16,
    alignItems: 'center',
  },
  timestamp: {
    fontSize: 11, // Tamaño de fuente más pequeño
    color: '#666', // Color más sutil
    marginTop: 2, // Espacio superior
    alignSelf: 'flex-end', // Alinear a la derecha dentro de la burbuja
    paddingHorizontal: 5, // Espaciado horizontal
  },
  suggestionsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap', // Permitir que las sugerencias se envuelvan en varias líneas
    marginTop: 8, // Espacio superior
  },
  suggestionButton: {
    backgroundColor: '#ddd', // Fondo gris claro para los botones de sugerencia
    borderRadius: 15, // Bordes redondeados
    paddingVertical: 6, // Relleno vertical
    paddingHorizontal: 12, // Relleno horizontal
    marginRight: 8, // Espacio a la derecha
    marginBottom: 8, // Espacio inferior
  },
  suggestionButtonText: {
    fontSize: 14, // Tamaño de fuente
    color: Colors.light.text, // Usar color de texto del tema
  },
  botContent: { // Nuevo estilo para el contenido del bot (icono + texto)
    flexDirection: 'row',
    alignItems: 'flex-start',
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
        // Guardar las sugerencias si existen en la respuesta
        suggestions: (typeof responseData === 'object' && responseData !== null && 'suggestions' in responseData && Array.isArray(responseData.suggestions))
          ? responseData.suggestions as string[]
          : [],
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
    // const botContentStyle = { flexDirection: 'row', alignItems: 'flex-start' }; // Estilo para el contenido del bot (icono + texto) - Eliminado

    // Determinar el icono basado en la intención
    let iconName: keyof typeof MaterialIcons.glyphMap | null = null;
    let iconColor = Colors.light.text; // Color por defecto

    if (msg.sender === 'bot' && msg.intent) {
      switch (msg.intent) {
        case 'greeting':
          iconName = 'waving-hand';
          break;
        case 'farewell':
          iconName = 'waving-hand'; // O 'logout', 'exit-to-app'
          break;
        case 'create_reminder':
          iconName = 'alarm-add';
          break;
        case 'list_reminders':
          iconName = 'list-alt';
          break;
        case 'date_query':
          iconName = 'calendar-today';
          break;
        case 'help':
          iconName = 'help-outline';
          break;
        // Puedes añadir más casos para otras intenciones
        default:
          iconName = 'chat-bubble-outline'; // Icono por defecto para bot
      }
    }


    return (
      <View
        key={msg.id}
        style={[
          messageStyle,
          { alignSelf: isUser ? 'flex-end' : 'flex-start' },
        ]}
      >
        {isUser ? (
          // Contenido para mensajes de usuario
          <ThemedText style={textStyle}>{msg.text}</ThemedText>
        ) : (
          // Contenido para mensajes del bot con icono opcional
          <View style={styles.botContent}>
            {iconName && (
              <MaterialIcons
                name={iconName}
                size={20}
                color={iconColor}
                style={{ marginRight: 8, marginTop: 2 }} // Espacio entre icono y texto
              />
            )}
            <ThemedText style={[textStyle, { flexShrink: 1 }]}>{msg.text}</ThemedText>
          </View>
        )}
        
        <ThemedText style={styles.timestamp}>
          {formatTime(msg.timestamp)}
        </ThemedText>
        {/* Mostrar sugerencias si existen y el mensaje es del bot */}
        {msg.sender === 'bot' && msg.suggestions && msg.suggestions.length > 0 && (
          <View style={styles.suggestionsContainer}>
            {msg.suggestions.map((suggestion, index) => (
              <TouchableOpacity
                key={index}
                style={styles.suggestionButton}
                onPress={() => {
                  // Al hacer clic en una sugerencia, establecer el texto en el input
                  setMessage(suggestion);
                  // Opcionalmente, enviar el mensaje automáticamente
                  // sendMessage();
                }}
              >
                <ThemedText style={styles.suggestionButtonText}>{suggestion}</ThemedText>
              </TouchableOpacity>
            ))}
          </View>
        )}
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
            placeholderTextColor="#999" // Color de placeholder más claro
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
