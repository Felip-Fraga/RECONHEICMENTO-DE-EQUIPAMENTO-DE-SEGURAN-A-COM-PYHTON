CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

MODEL_URL = "https://github.com/niconielsen32/ComputerVision/raw/master/YOLOv8-custom-trained/best.pt"
MODEL_PATH = "ppe_model.pt"

CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD = 0.45

CLASS_NAMES = {
    0:  "Capacete",
    1:  "Sem Capacete",
    2:  "Pessoa",
    3:  "Colete",
    4:  "Sem Colete",
    5:  "Luvas",
    6:  "Sem Luvas",
    7:  "Oculos",
    8:  "Sem Oculos",
    9:  "Botina",
    10: "Sem Botina",
    11: "Cinto de Seguranca",
    12: "Sem Cinto de Seguranca",
}

SAFE_CLASSES   = {0, 3, 5, 7, 9, 11}   # EPIs presentes
UNSAFE_CLASSES = {1, 4, 6, 8, 10, 12}  # EPIs ausentes

COLOR_SAFE     = (0, 255, 0)      # Verde
COLOR_UNSAFE   = (0, 0, 255)      # Vermelho
COLOR_PERSON   = (255, 165, 0)    # Laranja
COLOR_TEXT_BG  = (0, 0, 0)        # Preto

VIOLATIONS_DB_PATH = "violations.json"

VIOLATION_COOLDOWN_SECONDS = 10

FONT_SCALE      = 0.6
FONT_THICKNESS  = 2
BOX_THICKNESS   = 2