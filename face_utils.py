# face_utils.py  (LBPH)
import os
import json
import cv2
import numpy as np
from typing import Optional, Tuple

MODEL_FILE = "lbph_model.yml"
LABEL_MAP_FILE = "lbph_label_map.json"
FACE_SIZE = (160, 160)
RECOG_CONF_THRESHOLD = 70.0  # LBPH retorna 'confidence' (quanto menor melhor). Ajuste conforme necessário.

def load_face_db(model_file: str = MODEL_FILE, label_map_file: str = LABEL_MAP_FILE):
    """
    Retorna dict com {'recognizer': recognizer, 'label_map': label_map}
    Ou {} se nao encontrar arquivos.
    """
    if not os.path.exists(model_file) or not os.path.exists(label_map_file):
        print("[WARN] Modelo LBPH ou label_map nao encontrado. Rode lbph_train.py primeiro.")
        return {}

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except Exception as e:
        print("[ERROR] cv2.face nao disponivel. Instale opencv-contrib-python.")
        return {}

    recognizer.read(model_file)
    with open(label_map_file, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    # map labels back to int->name
    label_map = {int(k): v for k, v in label_map.items()}
    return {"recognizer": recognizer, "label_map": label_map}

def crop_face_from_bbox(frame: "np.ndarray", bbox: tuple) -> Optional["np.ndarray"]:
    """
    bbox: (x1,y1,x2,y2) — recorta a regiao superior da bbox e retorna face grayscale redimensionada.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w - 1, x2); y2 = min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    height = y2 - y1
    top_h = max(10, int(height * 0.5))
    ny1 = y1
    ny2 = min(y1 + top_h, y2)
    nx1 = x1
    nx2 = x2
    face = frame[ny1:ny2, nx1:nx2]
    if face.size == 0:
        return None
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    try:
        resized = cv2.resize(gray, FACE_SIZE)
    except Exception:
        return None
    return resized

def identify_face(face_img: "np.ndarray", db: dict) -> Tuple[Optional[str], float]:
    """
    face_img: imagem grayscale (FACE_SIZE) ou BGR — aceita ambos.
    db: object retornado por load_face_db
    Retorna (name, confidence) — name==None se desconhecido.
    """
    if not db:
        return None, 999.0
    recognizer = db.get("recognizer")
    label_map = db.get("label_map", {})
    if recognizer is None:
        return None, 999.0

    # face_img esperado grayscale; se BGR, converte
    img = face_img
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        label, conf = recognizer.predict(img)
    except Exception:
        return None, 999.0
    # LBPH: menor conf → melhor correspondência
    if conf <= RECOG_CONF_THRESHOLD:
        name = label_map.get(label, None)
        return name, float(conf)
    return None, float(conf)