import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time


mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
pyautogui.FAILSAFE = False #para evitar que se salga programa cuando salga de rango

Click_Ejecutado = False

Color_Puntero = (155, 0, 255)
                
anchocam, altocam = 640, 480 #valores de camara
#cuadro = 100  #Rango donde podemos interactuar con camara
Cuadro_InteraccionX = 270 # cuando mas grande mas chico el recuadro
Cuadro_InteraccionY = 190

anchopanta, altopanta = pyautogui.size() #Obtenemos las dimensiones de nuestra pantalla
print(anchopanta,altopanta)
sua = 3 #suavizado mouse
UbicAntex, UbicAntey = 0,0 # ubica puntos x e y anterior puntero
UbicActualx, UbicActualy = 0,0 # ubica puntos x e y actual

######################Lectura de la Camara

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3,anchocam) #definimos valores de camara
cap.set(4,altocam) 

#################### Declaramos variables para detectar boca abierta y calcular distancias

def Calculo_Distancia(x1, y1, x2, y2): #definimos funcion para calcular distancia entre 2 puntos
    p1 = np.array([x1, y1]) #punto 1 usando array de libreria numpy
    p2 = np.array([x2, y2]) #punto 2 usando array de libreria numpy
    return np.linalg.norm(p1 - p2) # retorna la distancia de puntos usando numpy


def Detecta_Movimientos(face_landmarks): #definimos esta funcion para detectar movimientos

    global Click_Ejecutado #declaro mi variable global 
    global Cuadro_InteraccionX
    
    #-----------------------Para Boca
    Boca_Abierta = False # para boca abierta falso colores de abajo
    color_vertical = (255, 0, 112) #color vertical boca
    color_horizontal = (255, 198, 82) #color horizontal boca
    
    x_Inferior_Labio = int(face_landmarks.landmark[17].x * width) 
    y_Inferior_Labio = int(face_landmarks.landmark[17].y * height) 

    x_Superior_Labio = int(face_landmarks.landmark[0].x * width) 
    y_Superior_Labio = int(face_landmarks.landmark[0].y * height)

    x_Izquierdo_Labio = int(face_landmarks.landmark[61].x * width) 
    y_Izquierdo_Labio = int(face_landmarks.landmark[61].y * height)

    x_Derecho_Labio = int(face_landmarks.landmark[291].x * width) 
    y_Derecho_Labio = int(face_landmarks.landmark[291].y * height)

    ################Calculo Longitud entre ojos

    color_distancia = (155, 30, 0)

    x_Punto_Ojo_Izq = int(face_landmarks.landmark[33].x * width) 
    y_Punto_Ojo_Izq = int(face_landmarks.landmark[33].y * height)

    x_Punto_Ojo_Der = int(face_landmarks.landmark[263].x * width) 
    y_Punto_Ojo_Der = int(face_landmarks.landmark[263].y * height)

    Distancia_Ojos = Calculo_Distancia(x_Punto_Ojo_Izq, y_Punto_Ojo_Izq, x_Punto_Ojo_Der, y_Punto_Ojo_Der)
    cv2.line(frame, (x_Punto_Ojo_Izq, y_Punto_Ojo_Izq), (x_Punto_Ojo_Der, y_Punto_Ojo_Der), color_distancia, 1)

    #Cuadro_InteraccionX = Distancia_Ojos*2

    print('Distancia_Ojos= ', Distancia_Ojos)

    #########################################

    Distancia_Horizontal = Calculo_Distancia(x_Izquierdo_Labio, y_Izquierdo_Labio, x_Derecho_Labio, y_Derecho_Labio) #distancia horizontal
    Distancia_Vertical = Calculo_Distancia(x_Inferior_Labio, y_Inferior_Labio, x_Superior_Labio, y_Superior_Labio)  #distancia vertical


    Relacion = Distancia_Horizontal/Distancia_Vertical
    print('Relacion = ', Relacion)

    if Relacion < 1.25: 
        Boca_Abierta = True # para dedo abajo true colores de abajo
        color_vertical = (255, 0, 255)
        color_horizontal = (255, 0, 255)

    if Relacion > 2.5:
        Click_Ejecutado = False


    print ('Estado Click = ', Click_Ejecutado)

    
    cv2.circle(frame, (x_Izquierdo_Labio, y_Izquierdo_Labio), 3, color_horizontal, 1)# primer numero diametro exterior y segundo el interior
    cv2.circle(frame, (x_Derecho_Labio, y_Derecho_Labio), 3, color_horizontal, 1)

    cv2.circle(frame, (x_Superior_Labio, y_Superior_Labio), 3, color_vertical, 1)
    cv2.circle(frame, (x_Inferior_Labio, y_Inferior_Labio), 3, color_vertical, 1)
    
    cv2.line(frame, (x_Izquierdo_Labio, y_Izquierdo_Labio), (x_Derecho_Labio, y_Derecho_Labio), color_horizontal, 1) # grosor de linea
    cv2.line(frame, (x_Superior_Labio, y_Superior_Labio), (x_Inferior_Labio, y_Inferior_Labio), color_vertical, 1)

    return Boca_Abierta

'''

def Detecta_Ojos(face_landmarks): #definimos esta funcion para detectar movimientos
    
    #-----------------------Para Ojo Izquierdo

    color_vertical = (255, 0, 112) #color vertical boca
    color_horizontal = (255, 198, 82) #color horizontal boca

    x_Superior_Ojo_Izq = int(face_landmarks.landmark[66].x * width) 
    y_Superior_Ojo_Izq = int(face_landmarks.landmark[65].y * height) 

    x_Inferior_Ojo_Izq = int(face_landmarks.landmark[100].x * width) 
    y_Inferior_Ojo_Izq = int(face_landmarks.landmark[100].y * height) 

    x_Izquierdo_Ojo_Izq = int(face_landmarks.landmark[10].x * width) 
    y_Izquierdo_Ojo_Izq = int(face_landmarks.landmark[10].y * height)

    x_Derecho_Ojo_Izq = int(face_landmarks.landmark[8].x * width) 
    y_Derecho_Ojo_Izq = int(face_landmarks.landmark[8].y * height)

    Distancia_Horizontal_Ojo_Izq = Calculo_Distancia(x_Izquierdo_Ojo_Izq, y_Izquierdo_Ojo_Izq, x_Derecho_Ojo_Izq, y_Derecho_Ojo_Izq) #distancia horizontal
    Distancia_Vertical_Ojo_Izq = Calculo_Distancia(x_Inferior_Ojo_Izq, y_Inferior_Ojo_Izq, x_Superior_Ojo_Izq, y_Superior_Ojo_Izq)  #distancia vertical


    print('superior = ', Distancia_Vertical_Ojo_Izq)
    print('inferior = ', Distancia_Horizontal_Ojo_Izq)
    Relacion = Distancia_Vertical_Ojo_Izq/Distancia_Horizontal_Ojo_Izq

    if Relacion > 1.45:
        print('Señal de Ojo')
        color_vertical = (255, 0, 255)
        color_horizontal = (255, 0, 255)
        #winsound.PlaySound('SystemQuestion', winsound.SND_ALIAS)

    cv2.circle(frame, (x_Izquierdo_Ojo_Izq, y_Izquierdo_Ojo_Izq), 5, color_horizontal, 2)
    cv2.circle(frame, (x_Derecho_Ojo_Izq, y_Derecho_Ojo_Izq), 5, color_horizontal, 2)

    cv2.circle(frame, (x_Superior_Ojo_Izq, y_Superior_Ojo_Izq), 5, color_vertical, 2)
    cv2.circle(frame, (x_Inferior_Ojo_Izq, y_Inferior_Ojo_Izq), 5, color_vertical, 2)
    
    cv2.line(frame, (x_Izquierdo_Ojo_Izq, y_Izquierdo_Ojo_Izq), (x_Derecho_Ojo_Izq, y_Derecho_Ojo_Izq), color_horizontal, 3)
    cv2.line(frame, (x_Superior_Ojo_Izq, y_Superior_Ojo_Izq), (x_Inferior_Ojo_Izq, y_Inferior_Ojo_Izq), color_vertical, 3)
    
    print('superior = ', Distancia_Vertical_Ojo_Izq)
    print('inferior = ', Distancia_Horizontal_Ojo_Izq)
    Relacion = Distancia_Vertical_Ojo_Izq/Distancia_Horizontal_Ojo_Izq

    if Relacion > 1.33:
        print('Señal de Ojo')
        color_vertical = (255, 0, 255)
        color_horizontal = (255, 0, 255)
        #winsound.PlaySound('SystemQuestion', winsound.SND_ALIAS)

    
'''

    

################

with mp_face_mesh.FaceMesh(
    static_image_mode=False, # false para videos, true para imagenes
    max_num_faces=1, #solo detectamos una cara
    min_detection_confidence=0.5) as face_mesh: # valor confianza de deteccion

    while True:

        ret, frame = cap.read() #leemos la camara
        if ret == False:
            break

        height, width, _ = frame.shape
        frame = cv2.flip(frame,1) #detectamos la cara
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # la convertimos a RGB
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                
                cv2.rectangle(frame, (Cuadro_InteraccionX, Cuadro_InteraccionY), (anchocam - Cuadro_InteraccionX, altocam - Cuadro_InteraccionY), (100, 0, 0), 2)  # Generamos cuadro
                    
                x = int(face_landmarks.landmark[19].x * width)
                y = int(face_landmarks.landmark[19].y * height)

                cv2.circle(frame, (x, y), 6, Color_Puntero, 1)
                cv2.circle(frame, (x, y), 2, Color_Puntero, -1)

                #-----------------> Modo movimiento conversion a las pixeles de mi pantalla-------------
                x3 = np.interp(x, (Cuadro_InteraccionX,anchocam - Cuadro_InteraccionX), (0,anchopanta))
                y3 = np.interp(y, (Cuadro_InteraccionY, altocam - Cuadro_InteraccionY), (0, altopanta))

                #------------------------------- Suavizado los valores ----------------------------------
                UbicActualx = UbicAntex + (x3 - UbicAntex) / sua #Ubicacion actual = ubi anterior + x3 - Pa dividida el valor suavizado
                UbicActualy = UbicAntey + (y3 - UbicAntey) / sua

                print('UbicActualx=' ,UbicActualx, 'UbicActualy =' ,UbicActualy)

                #-------------------------------- Mover el Mouse ---------------------------------------
                #pyautogui.moveTo(UbicActualx ,UbicActualy) #Enviamos las coordenadas al Mouse
                #cv2.circle(frame, (x,y), 10, (0,0,0), cv2.FILLED)
                #UbicAntex, UbicAntey = UbicActualx, UbicActualy

                '''

                #print('x real igual =', x)
                if x<470:
                    x=0
                elif x>790:
                    x=1920
                else:
                    x = mapeox(x, 470, 790, 0, 1920)

                if y<245:
                    y=0
                elif y>485:
                    y=1080
                else:
                    y = mapeoy(y, 245, 485, 0, 1080)

                
                #print("x mapeado =", x)

                #pyautogui.moveTo(int(x*1), int(y*1)) 

                #print('x igual =', x)
                #print('y igual =', y)

                #x = int(face_landmarks.landmark[index].x * width)
                #y = int(face_landmarks.landmark[index].y * height)
                #cv2.circle(frame, (x, y), 2, (255, 0, 255), 2)
                '''
                #if Detecta_Ojos(face_landmarks):
                #    print('ojos')


                if Detecta_Movimientos(face_landmarks) and Click_Ejecutado==False:
                    pyautogui.click()
                    Click_Ejecutado=True
                    #winsound.PlaySound('SystemQuestion', winsound.SND_ALIAS)
                    print("click")

                #if Detecta_Movimientos(face_landmarks):
                    #pyautogui.moveTo(UbicActualx ,UbicActualy) #Enviamos las coordenadas al Mouse
                    #break #ejecuta mas rapido el movimiento

                #para separar las coordenas buscar la posicion actual del puntero
                #-------------------------------- Mover el Mouse ---------------------------------------
                #pyautogui.moveTo(UbicActualx ,UbicActualy) #Enviamos las coordenadas al Mouse
                #cv2.circle(frame, (x,y), 10, (0,0,0), cv2.FILLED)
                UbicAntex, UbicAntey = UbicActualx, UbicActualy
                break
                    
                    
                    

                    
        #time.sleep(1)
        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()