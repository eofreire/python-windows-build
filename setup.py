import sys
from cx_Freeze import setup, Executable

# Dependências
build_exe_options = {
    "packages": ["cv2", "mediapipe", "tensorflow", "numpy"],
    "includes": ["csv", "time", "os", "sys"],
    "include_files": [],
    "excludes": [],
    "build_exe": "./build"
}

# Configuração base para o executável
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name="BodyCapture",
    version="1.0",
    description="Body Capture Application",
    options={"build_exe": build_exe_options},
    executables=[
        Executable(
            "body_capture.py",
            base=base,
            target_name="BodyCapture.exe"
        )
    ]
)
