
import setuptools
import distutils
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
from datetime import datetime, timezone, timedelta
import ssl
import urllib.request

CAPTURE_FPS = 30.0    # Camera capture rate
PROCESS_FPS = 30.0    # Landmark detection rate
VIDEO_CODEC = 'mp4v'  # Codec for MP4 format
VIDEO_FORMAT = '.mp4'  # File extension

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context
urllib.request.urlopen = lambda url, *args, **kwargs: urllib.request.urlopen(url, *args, **kwargs, context=ssl._create_unverified_context())

# Configure environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def configure_mediapipe():
    """Configure MediaPipe to use local models"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mediapipe_path = os.path.dirname(mp.__file__)

    models = {
        'pose': {
            'source': os.path.join(current_dir, 'pose_landmark_lite.tflite'),
            'dest': os.path.join(mediapipe_path, 'modules', 'pose_landmark', 'pose_landmark_lite.tflite')
        },
        'hand': {
            'source': os.path.join(current_dir, 'hand_landmark.tflite'),
            'dest': os.path.join(mediapipe_path, 'modules', 'hand_landmark', 'hand_landmark.tflite')
        }
    }

    for model in models.values():
        os.makedirs(os.path.dirname(model['dest']), exist_ok=True)
        if not os.path.exists(model['dest']):
            if os.path.exists(model['source']):
                import shutil
                shutil.copy2(model['source'], model['dest'])
            else:
                raise FileNotFoundError(f"Model not found: {model['source']}")

    mp.solutions.pose._POSE_LANDMARK_MODEL_PATH = models['pose']['dest']
    mp.solutions.hands._HAND_LANDMARK_MODEL_PATH = models['hand']['dest']
    
def check_gpu():
    """Check for GPU availability"""
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print("GPU detected and enabled:", physical_devices)
    else:
        print("No GPU detected. Running on CPU.")

# Initialize MediaPipe and GPU
try:
    configure_mediapipe()
    check_gpu()
except Exception as e:
    print(f"Error during initialization: {e}")
    sys.exit(1)

# Initialize MediaPipe components
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

class MotionCaptureApp:
    """Application for capturing motion data using MediaPipe"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Motion Capture System")
        
        # Configure style
        self.style = ttk.Style()
        self.style.configure('Custom.TButton', padding=5)
        
        # Initialize variables
        self.filename = None
        self.recording = False
        self.cap = None
        self.save_video = tk.BooleanVar()
        self.record_fps = tk.DoubleVar(value=10.0)  # Default recording FPS
        self.canvas_width = 1280
        self.canvas_height = 720
        
        # Create GUI
        self.create_gui()
        self.drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

    def create_gui(self):
        # Main frame
        self.frame_controls = ttk.Frame(self.root)
        self.frame_controls.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # Canvas with specific size and black background
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg='black')
        self.canvas.pack(padx=10, pady=5, expand=True, fill=tk.BOTH)
        
        # Create other GUI elements
        self.create_control_buttons()
        self.create_status_display()

    def clear_canvas(self):
        """Clear the canvas and reset to black background"""
        self.canvas.delete("all")
        self.canvas.create_rectangle(
            0, 0,
            self.canvas_width, self.canvas_height,
            fill='black'
        )
        self.root.update()

    def create_control_buttons(self):
        self.select_button = ttk.Button(
            self.frame_controls,
            text="Select File",
            command=self.select_file,
            style='Custom.TButton'
        )
        self.select_button.pack(side=tk.LEFT, padx=5)
        
        self.start_button = ttk.Button(
            self.frame_controls,
            text="Start Capture",
            command=self.start_capture,
            state=tk.DISABLED,
            style='Custom.TButton'
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(
            self.frame_controls,
            text="Stop Capture",
            command=self.stop_capture,
            state=tk.DISABLED,
            style='Custom.TButton'
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.exit_button = ttk.Button(
            self.frame_controls,
            text="Exit",
            command=self.exit_application,
            style='Custom.TButton'
        )
        self.exit_button.pack(side=tk.RIGHT, padx=5)
        
        self.video_checkbox = ttk.Checkbutton(
            self.frame_controls,
            text="Save Video",
            variable=self.save_video
        )
        self.video_checkbox.pack(side=tk.LEFT, padx=5)
        
        # Add entry for recording FPS
        self.fps_label = ttk.Label(self.frame_controls, text="Recording FPS:")
        self.fps_label.pack(side=tk.LEFT, padx=5)
        
        self.fps_entry = ttk.Entry(self.frame_controls, textvariable=self.record_fps, width=5)
        self.fps_entry.pack(side=tk.LEFT, padx=5)

    def create_status_display(self):
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(
            self.root, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        self.status_var.set("Ready to start")
        
        self.timestamp_var = tk.StringVar()
        self.timestamp_label = ttk.Label(
            self.root,
            textvariable=self.timestamp_var,
            anchor=tk.E
        )
        self.timestamp_label.pack(side=tk.BOTTOM, fill=tk.X, padx=10)
        self.update_timestamp()

    def update_timestamp(self):
        current_time = datetime.now(timezone(timedelta(hours=-3))).strftime('%Y-%m-%d %H:%M:%S (GMT-3)')
        self.timestamp_var.set(f"Current time: {current_time}")
        self.root.after(1000, self.update_timestamp)

    def select_file(self):
        current_time = datetime.now(timezone(timedelta(hours=-3))).strftime('%Y%m%d_%H%M%S')
        username = os.getlogin()
        default_filename = f"motion_capture_{username}_{current_time}.csv"
        
        self.filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Save data file",
            initialfile=default_filename
        )
        if self.filename:
            self.start_button.config(state=tk.NORMAL)
            self.status_var.set(f"Selected file: {os.path.basename(self.filename)}")

    def start_capture(self):
        if not self.filename:
            return
        
        self.recording = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.select_button.config(state=tk.DISABLED)
        
        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
            
            if not self.cap.isOpened():
                raise Exception("Unable to access camera")
            
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"Camera initialized at {actual_fps} FPS")
            
            self.status_var.set("Capturing motion data...")
            self.capture_data()
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.stop_capture()
        
    def capture_data(self):
        video_writer = None
        username = os.getlogin()  # Get current user's login
        target_fps = self.record_fps.get()  # Get user-defined recording FPS
        frame_interval = 1.0 / target_fps
        
        if self.save_video.get():
            video_filename = self.filename.replace(".csv", VIDEO_FORMAT)
            
            # Initialize video writer with user-defined FPS
            codecs_to_try = [
                ('avc1', 'H.264'),
                ('mp4v', 'MP4V'),
                ('X264', 'H.264'),
                ('XVID', 'XVID')
            ]
            
            for codec, codec_name in codecs_to_try:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    ret, frame = self.cap.read()
                    if ret:
                        height, width = frame.shape[:2]
                        test_writer = cv2.VideoWriter(
                            video_filename,
                            fourcc,
                            target_fps,  # Use user-defined FPS
                            (width, height)
                        )
                        
                        if test_writer.isOpened():
                            video_writer = test_writer
                            print(f"User {username} recording video using {codec_name} codec at {target_fps} FPS")
                            break
                        else:
                            test_writer.release()
                except Exception as e:
                    print(f"Failed to initialize {codec_name} codec: {str(e)}")

        # Initialize timing variables
        start_time = time.time()
        last_frame_time = start_time
        frames_processed = 0
        frames_recorded = 0
        
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write headers...
            headers = ['timestamp', 'elapsed_time', 'width', 'height']
            for i in range(33):
                headers += [f'pose_landmark_{i}_x', f'pose_landmark_{i}_y', f'pose_landmark_{i}_z', f'pose_landmark_{i}_visibility']
            for i in range(21):
                headers += [f'left_hand_landmark_{i}_x', f'left_hand_landmark_{i}_y', f'left_hand_landmark_{i}_z']
                headers += [f'right_hand_landmark_{i}_x', f'right_hand_landmark_{i}_y', f'right_hand_landmark_{i}_z']
            writer.writerow(headers)
            
            with mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=0
            ) as holistic:
                while self.recording and self.cap and self.cap.isOpened():
                    current_time = time.time()
                    elapsed_time = current_time - last_frame_time
                    
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    
                    # Process frame with landmarks
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image)
                    
                    # Prepare display frame
                    display_image = frame.copy()
                    self.draw_landmarks(display_image, results)
                    
                    # Calculate real FPS and recording FPS
                    frames_processed += 1
                    process_time = current_time - start_time
                    real_fps = frames_processed / process_time
                    recording_fps = frames_recorded / process_time if frames_recorded > 0 else 0
                    
                    # Record video frame at controlled rate
                    if video_writer and elapsed_time >= frame_interval:
                        video_writer.write(display_image)
                        frames_recorded += 1
                        last_frame_time = current_time
                    
                    # Update display
                    self.display_frame(display_image)
                    
                    # Get timestamp in UTC format
                    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                    height, width, _ = frame.shape
                    
                    if results.pose_landmarks or results.left_hand_landmarks or results.right_hand_landmarks:
                        landmarks = self.process_landmarks(results, timestamp, process_time, width, height)
                        writer.writerow(landmarks)
                    
                    self.status_var.set(
                        f"User: {username} | Processing: {frames_processed} frames ({real_fps:.1f} FPS) | "
                        f"Recording: {frames_recorded} frames ({recording_fps:.1f} FPS)"
                    )
                                        
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC
                        break
            
            if video_writer:
                video_writer.release()

    def stop_capture(self):
        """Stop capture and clear display"""
        self.recording = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.select_button.config(state=tk.NORMAL)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Clear the display
        self.clear_canvas()
        self.status_var.set("Capture finished")

    def draw_landmarks(self, image, results):
        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=self.drawing_spec
        )
        # Draw hand landmarks
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=self.drawing_spec
        )
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=self.drawing_spec
        )

    def process_landmarks(self, results, current_time, elapsed_time, width, height):
        landmarks = [current_time, elapsed_time, width, height]
        
        # Process pose landmarks
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        else:
            landmarks.extend([0] * (33 * 4))  # 33 landmarks with x, y, z, visibility
        
        # Process hand landmarks
        for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
            else:
                landmarks.extend([0] * (21 * 3))  # 21 landmarks with x, y, z
        
        return landmarks

    def display_frame(self, frame):
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # Calculate scale while maintaining aspect ratio
        scale = min(canvas_width/frame_width, canvas_height/frame_height)
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)
        
        # Resize frame
        if new_width > 0 and new_height > 0:
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Convert colors and create image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Calculate center position
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        
        # Clear canvas and draw new image
        self.canvas.delete("all")
        self.canvas.create_image(x, y, anchor=tk.NW, image=imgtk)
        self.canvas.image = imgtk
        self.root.update()

    def exit_application(self):
        if self.cap:
            self.cap.release()
        self.root.quit()
        self.root.destroy()

def main():
    try:
        root = tk.Tk()
        app = MotionCaptureApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")

if __name__ == "__main__":
    main()
