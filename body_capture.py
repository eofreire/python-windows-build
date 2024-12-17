import cv2
import mediapipe as mp
import csv
import time

# Inicializar MediaPipe Pose con modelos locales
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_path='pose_landmark_lite.tflite'  # Ruta local al modelo .tflite
)

# Inicializar MediaPipe Hands con el modelo local
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_path='hand_landmark.tflite'  # Ruta local al modelo .tflite
)

# Configuración de captura de video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

# Variável para armazenar o timestamp do início
start_time = None

# Abrir arquivo CSV para salvar os landmarks
with open('landmarks.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Escribir los encabezados del CSV
    headers = ['elapsed_time']  # Añadir el tiempo transcurrido como el primer encabezado
    for i in range(33):
        headers += [f'pose_landmark_{i}_x', f'pose_landmark_{i}_y', f'pose_landmark_{i}_z', f'pose_landmark_{i}_visibility']
    for i in range(21):
        headers += [f'left_hand_landmark_{i}_x', f'left_hand_landmark_{i}_y', f'left_hand_landmark_{i}_z']
        headers += [f'right_hand_landmark_{i}_x', f'right_hand_landmark_{i}_y', f'right_hand_landmark_{i}_z']
    writer.writerow(headers)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Capturar el timestamp actual
        current_timestamp = time.time()

        # Definir el timestamp inicial
        if start_time is None:
            start_time = current_timestamp

        # Calcular el tiempo transcurrido desde el inicio
        elapsed_time = current_timestamp - start_time

        # Convertir la imagen a RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar la pose
        pose_results = pose.process(image)

        # Procesar las manos
        hand_results = hands.process(image)

        # Dibujar los landmarks en la imagen
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if pose_results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        # Mostrar la imagen
        cv2.imshow('MediaPipe Local Models', image)

        # Guardar los landmarks y el tiempo transcurrido en el archivo CSV
        if pose_results.pose_landmarks or hand_results.multi_hand_landmarks:
            landmarks = [elapsed_time]  # Añadir el tiempo transcurrido como el primer dato
            if pose_results.pose_landmarks:
                for landmark in pose_results.pose_landmarks.landmark:
                    landmarks += [landmark.x, landmark.y, landmark.z, landmark.visibility]
            else:
                landmarks += [None] * 132  # 33 landmarks * 4 valores (x, y, z, visibility)
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        landmarks += [landmark.x, landmark.y, landmark.z]
            else:
                landmarks += [None] * 63  # 21 landmarks * 3 valores (x, y, z)
            writer.writerow(landmarks)

        # Finalizar si se presiona ESC
        if cv2.waitKey(10) & 0xFF == 27:
            break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
