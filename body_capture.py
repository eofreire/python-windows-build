import mediapipe as mp
import cv2
import time
import tensorflow as tf

# Cargar los modelos de TensorFlow Lite
pose_model = tf.lite.Interpreter(model_path="pose_landmark_lite.tflite")
hand_model = tf.lite.Interpreter(model_path="hand_landmark.tflite")

# Inicializar la sesión de ejecución
pose_model.allocate_tensors()
hand_model.allocate_tensors()

# Función para ejecutar el modelo de pose
def run_pose_model(image):
    # Preprocesar la imagen
    input_details = pose_model.get_input_details()
    output_details = pose_model.get_output_details()

    input_data = image.astype('float32')
    pose_model.set_tensor(input_details[0]['index'], input_data)
    pose_model.invoke()

    # Obtener los resultados
    output_data = pose_model.get_tensor(output_details[0]['index'])
    return output_data

# Función para ejecutar el modelo de manos
def run_hand_model(image):
    input_details = hand_model.get_input_details()
    output_details = hand_model.get_output_details()

    input_data = image.astype('float32')
    hand_model.set_tensor(input_details[0]['index'], input_data)
    hand_model.invoke()

    # Obtener los resultados
    output_data = hand_model.get_tensor(output_details[0]['index'])
    return output_data

# Configuración de captura de video
cap = cv2.VideoCapture(0)
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen a RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Ejecutar los modelos de pose y manos
    pose_results = run_pose_model(image)
    hand_results = run_hand_model(image)

    # Visualizar resultados (solo como referencia)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Mostrar ventana
    cv2.imshow('MediaPipe Local Models', image)

    # Salir con tecla ESC
    if cv2.waitKey(10) & 0xFF == 27:
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
