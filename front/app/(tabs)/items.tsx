import React, { useState, useEffect } from 'react';
import { StyleSheet, FlatList, View, Text, ActivityIndicator, TouchableOpacity, Alert } from 'react-native'; // Importar TouchableOpacity y Alert
import { ThemedView } from '@components/ThemedView';
import { ThemedText } from '@components/ThemedText';
import { API_URL } from '@constants/Api'; // Asumiendo que la API_URL es accesible aquí
import { MaterialIcons } from '@expo/vector-icons'; // Importar MaterialIcons

// Definir interfaces para los datos del backend
interface ReminderData {
  id: string;
  texto: string; // El backend usa 'texto' para recordatorios
  fecha: string; // Fecha en formato ISO 8601
  recurrente: boolean;
  intervalo?: string;
  max_repeticiones?: number;
  repeticiones: number;
  activo: boolean;
}

interface EventData {
  id: string;
  titulo: string; // El backend usa 'titulo' para eventos
  fecha: string; // Fecha en formato ISO 8601
  ubicacion?: string;
  descripcion?: string;
  activo: boolean;
}

// Definir tipo para los items combinados en el frontend
interface Item {
  id: string;
  text?: string; // Para recordatorios (mapeado de 'texto')
  title?: string; // Para eventos (mapeado de 'titulo')
  date: string; // Fecha en formato ISO 8601 (mapeado de 'fecha')
  type: 'reminder' | 'event'; // Para distinguir entre recordatorios y eventos
  // Incluir otros campos relevantes si se necesitan mostrar
  ubicacion?: string;
  descripcion?: string;
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 16,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  itemContainer: {
    backgroundColor: '#fff',
    padding: 16,
    marginBottom: 12,
    borderRadius: 8,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.2,
    shadowRadius: 2,
  },
  itemTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  itemDate: {
    fontSize: 14,
    color: '#666',
  },
  noItemsText: {
    fontSize: 18,
    textAlign: 'center',
    marginTop: 20,
    color: '#888',
  },
  itemContent: {
    flexDirection: 'row', // Alinear contenido y botón horizontalmente
    justifyContent: 'space-between', // Espacio entre contenido y botón
    alignItems: 'center', // Centrar verticalmente
  },
  itemTextContainer: {
    flex: 1, // Permitir que el texto ocupe el espacio disponible
    marginRight: 10, // Espacio a la derecha del texto antes del botón
  },
  itemDetail: {
    fontSize: 14,
    color: '#555',
    marginTop: 2,
  },
});

export default function ItemsScreen() {
  const [items, setItems] = useState<Item[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchItems();
  }, []);

  const fetchItems = async () => {
    setIsLoading(true);
    try {
      // Asumiendo que hay un endpoint en el backend para obtener recordatorios y eventos
      // Podríamos necesitar endpoints separados o uno combinado
      const remindersResponse = await fetch(`${API_URL}/reminders`); // Ejemplo de endpoint para recordatorios
      const eventsResponse = await fetch(`${API_URL}/events`); // Ejemplo de endpoint para eventos

      if (!remindersResponse.ok || !eventsResponse.ok) {
        throw new Error('Error al obtener los items');
      }

      const remindersData = await remindersResponse.json();
      const eventsData = await eventsResponse.json();

      // Combinar y formatear los datos, mapeando los campos del backend a la interfaz Item
      const loadedItems: Item[] = [
        ...(remindersData as ReminderData[]).map(r => ({
          id: r.id,
          text: r.texto,
          date: r.fecha,
          type: 'reminder' as 'reminder' // Especificar el tipo explícitamente
        })),
        ...(eventsData as EventData[]).map(e => ({
          id: e.id,
          title: e.titulo,
          date: e.fecha,
          type: 'event' as 'event', // Especificar el tipo explícitamente
          ubicacion: e.ubicacion, // Incluir ubicación si existe
          descripcion: e.descripcion, // Incluir descripción si existe
        })),
      ];

      // Ordenar por fecha
      loadedItems.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());

      setItems(loadedItems);
    } catch (error) {
      console.error('Error fetching items:', error);
      // Mostrar un mensaje de error al usuario si es necesario
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteItem = async (itemToDelete: Item) => {
    try {
      const endpoint = itemToDelete.type === 'reminder' ? `${API_URL}/reminders/${itemToDelete.id}` : `${API_URL}/events/${itemToDelete.id}`;
      
      const response = await fetch(endpoint, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error(`Error al eliminar ${itemToDelete.type}: ${response.statusText}`);
      }

      // Actualizar la lista de items eliminando el item borrado
      setItems(prevItems => prevItems.filter(item => item.id !== itemToDelete.id));
      console.log(`${itemToDelete.type} con ID ${itemToDelete.id} eliminado exitosamente.`);

    } catch (error) {
      console.error('Error deleting item:', error);
      // Mostrar un mensaje de error al usuario si es necesario
      Alert.alert('Error', `No se pudo eliminar el ${itemToDelete.type}.`); // Usar Alert.alert
    }
  };

  const renderItem = ({ item }: { item: Item }) => (
    <View style={styles.itemContainer}>
      <View style={styles.itemContent}> {/* Contenedor para el contenido y el botón */}
        <View style={styles.itemTextContainer}> {/* Contenedor para el texto */}
          <ThemedText style={styles.itemTitle}>
            {item.type === 'reminder' ? item.text : item.title}
          </ThemedText>
          <ThemedText style={styles.itemDate}>
            Fecha: {new Date(item.date).toLocaleString()} {/* Formatear la fecha */}
          </ThemedText>
          {/* Podríamos añadir más detalles aquí (ubicación, descripción, etc.) */}
          {item.type === 'event' && item.ubicacion && (
            <ThemedText style={styles.itemDetail}>Ubicación: {item.ubicacion}</ThemedText>
          )}
           {item.type === 'event' && item.descripcion && (
            <ThemedText style={styles.itemDetail}>Descripción: {item.descripcion}</ThemedText>
          )}
        </View>
        <TouchableOpacity onPress={() => handleDeleteItem(item)}>
           {/* Usar un icono de MaterialIcons para eliminar */}
          <MaterialIcons name="delete" size={24} color="#ff0000" />
        </TouchableOpacity>
      </View>
    </View>
  );

  if (isLoading) {
    return (
      <ThemedView style={styles.loadingContainer}>
        <ActivityIndicator size="large" />
        <ThemedText>Cargando items...</ThemedText>
      </ThemedView>
    );
  }

  return (
    <ThemedView style={styles.container}>
      <ThemedText style={styles.title}>Mis Items</ThemedText>
      {items.length === 0 ? (
        <ThemedText style={styles.noItemsText}>No tienes recordatorios o eventos próximos.</ThemedText>
      ) : (
        <FlatList
          data={items}
          renderItem={renderItem}
          keyExtractor={(item) => item.id}
        />
      )}
    </ThemedView>
  );
}