import cv2
import mediapipe as mp
import csv
import time
import os
import sys
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from mediapipe.framework.formats import rect_pb2

# Configure environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow log messages
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Enable GPU

def check_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print("GPU detected and enabled:", physical_devices)
    else:
        print("No GPU detected. Running on CPU.")

check_gpu()

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

class DataCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Capture - MediaPipe")
        
        self.filename = None
        self.recording = False
        self.cap = None
        self.save_video = tk.BooleanVar()
        
        # Create main layout
        self.frame_controls = ttk.Frame(root)
        self.frame_controls.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()
        
        self.select_button = ttk.Button(self.frame_controls, text="Select File", command=self.select_file)
        self.select_button.pack(side=tk.LEFT, padx=5)
        
        self.start_button = ttk.Button(self.frame_controls, text="Start Capture", command=self.start_capture, state=tk.DISABLED)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(self.frame_controls, text="Stop Capture", command=self.stop_capture, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.video_checkbox = ttk.Checkbutton(self.frame_controls, text="Save Video", variable=self.save_video)
        self.video_checkbox.pack(side=tk.LEFT, padx=5)
    
    def select_file(self):
        self.filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if self.filename:
            self.start_button.config(state=tk.NORMAL)
    
    def start_capture(self):
        if not self.filename:
            return
        self.recording = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.capture_data()
    
    def stop_capture(self):
        self.recording = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        if self.cap:
            self.cap.release()
    
    def capture_data(self):
        video_writer = None
        if self.save_video.get():
            video_filename = self.filename.replace(".csv", ".avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(video_filename, fourcc, 30.0, (640, 480))
        
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            headers = ['elapsed_time', 'width', 'height']
            for i in range(33):
                headers += [f'pose_landmark_{i}_x', f'pose_landmark_{i}_y', f'pose_landmark_{i}_z', f'pose_landmark_{i}_visibility']
            for i in range(21):
                headers += [f'left_hand_landmark_{i}_x', f'left_hand_landmark_{i}_y', f'left_hand_landmark_{i}_z']
                headers += [f'right_hand_landmark_{i}_x', f'right_hand_landmark_{i}_y', f'right_hand_landmark_{i}_z']
            writer.writerow(headers)
            
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0) as holistic:
                start_time = time.time()
                
                while self.recording and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    
                    elapsed_time = time.time() - start_time
                    height, width, _ = frame.shape
                    
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    
                    self.display_frame(image)
                    
                    if video_writer:
                        video_writer.write(image)
                    
                    if results.pose_landmarks or results.left_hand_landmarks or results.right_hand_landmarks:
                        landmarks = [elapsed_time, width, height]
                        if results.pose_landmarks:
                            for landmark in results.pose_landmarks.landmark:
                                landmarks += [landmark.x * width, landmark.y * height, landmark.z, landmark.visibility]
                        else:
                            landmarks += [float('nan')] * 132
                        if results.left_hand_landmarks:
                            for landmark in results.left_hand_landmarks.landmark:
                                landmarks += [landmark.x * width, landmark.y * height, landmark.z]
                        else:
                            landmarks += [float('nan')] * 63
                        if results.right_hand_landmarks:
                            for landmark in results.right_hand_landmarks.landmark:
                                landmarks += [landmark.x * width, landmark.y * height, landmark.z]
                        else:
                            landmarks += [float('nan')] * 63
                        writer.writerow(landmarks)
                    
                    if cv2.waitKey(10) & 0xFF == 27:
                        break
                
        if video_writer:
            video_writer.release()
        self.cap.release()
        self.cap = None

    def display_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.root.update()

if __name__ == "__main__":
    root = tk.Tk()
    app = DataCaptureApp(root)
    root.mainloop()
