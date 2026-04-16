"""
====
  EPI Monitor — Sistema Completo de Detecção de EPI
====
  Funcionalidades:
    - Detecção de capacete + colete via YOLOv8 customizado
    - Tracking real com ID fixo por pessoa (IoU)
    - Integração com Firebase Firestore
    - Liberação/bloqueio automático por RFID (simulado via teclado)
    - GPU (CUDA) com FP16 para máximo FPS
    - Fallback automático para CPU
    - Logging JSON + TXT

  Dependências:
    pip install opencv-python ultralytics numpy firebase-admin torch torchvision

  Uso:
    python datasetpropriov2.py
====
"""

import cv2
import json
import time
import threading
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from collections import deque

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[AVISO] ultralytics nao instalado. Execute: pip install ultralytics")

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    print("[AVISO] firebase-admin nao instalado. Firebase desativado.")


# ====
#  CONFIGURAÇÕES GLOBAIS
# ====

CONFIG = {
    # Câmera
    "camera_index":   0,
    "frame_width":    1280,
    "frame_height":   720,

    # Modelo
    "model_path": "yolov8n.pt",   # seu modelo customizado
    "device":         "cuda" if torch.cuda.is_available() else "cpu",
    "half":           True,                   # FP16 — so ativo com GPU
    "infer_size":     640,                    # 512 = mais FPS, 640 = mais precisao
    "conf":           0.40,
    "iou":            0.50,

    # Tracking
    "max_disappeared": 30,
    "iou_threshold":   0.35,

    # Firebase
    "firebase_key":   "firebase_key.json",
    "firebase_col":   "epi_logs",

    # Logs locais
    "log_dir":        Path("logs"),
    "json_interval":  30,

    # Regras de EPI
    "require_helmet": True,
    "require_vest":   True,

    # Cooldown de log por pessoa (segundos)
    "log_cooldown":   3.0,

    # Cores BGR
    "color_ok":       (0, 200, 0),
    "color_danger":   (0, 0, 220),
    "color_warn":     (0, 165, 255),
    "color_white":    (255, 255, 255),
    "color_yellow":   (0, 220, 220),
}

# Nomes de classe do modelo customizado
HELMET_LABELS = {"helmet", "capacete", "hard hat", "hardhat", "safety helmet"}
VEST_LABELS   = {"vest", "colete", "safety vest", "reflective vest", "high-vis vest"}
PERSON_LABELS = {"person", "pessoa"}


# ====
#  NUMPY JSON ENCODER
# ====

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


# ====
#  TRACKER — ID FIXO POR PESSOA
# ====

class PersonTracker:
    def __init__(self):
        self.next_id = 1
        self.tracks: dict = {}

    def update(self, detections: list) -> list:
        if not detections:
            for tid in list(self.tracks):
                self.tracks[tid]["disappeared"] += 1
                if self.tracks[tid]["disappeared"] > CONFIG["max_disappeared"]:
                    del self.tracks[tid]
            return []

        det_bboxes = [d["bbox"] for d in detections]

        if not self.tracks:
            for det in detections:
                self._register(det)
            return self._assign_ids(detections)

        track_ids    = list(self.tracks.keys())
        track_bboxes = [self.tracks[tid]["bbox"] for tid in track_ids]

        iou_matrix = np.zeros((len(track_ids), len(det_bboxes)))
        for i, tb in enumerate(track_bboxes):
            for j, db in enumerate(det_bboxes):
                iou_matrix[i, j] = self._iou(tb, db)

        matched_tracks = set()
        matched_dets   = set()

        while True:
            if iou_matrix.size == 0:
                break
            idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            i, j = idx
            if iou_matrix[i, j] < CONFIG["iou_threshold"]:
                break
            tid = track_ids[i]
            self.tracks[tid]["bbox"]        = det_bboxes[j]
            self.tracks[tid]["disappeared"] = 0
            detections[j]["track_id"]       = tid
            matched_tracks.add(i)
            matched_dets.add(j)
            iou_matrix[i, :] = -1
            iou_matrix[:, j] = -1

        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                self.tracks[tid]["disappeared"] += 1
                if self.tracks[tid]["disappeared"] > CONFIG["max_disappeared"]:
                    del self.tracks[tid]

        for j, det in enumerate(detections):
            if j not in matched_dets:
                self._register(det)

        return self._assign_ids(detections)

    def _register(self, det: dict):
        tid = self.next_id
        self.tracks[tid] = {"bbox": det["bbox"], "disappeared": 0}
        det["track_id"]  = tid
        self.next_id    += 1

    @staticmethod
    def _assign_ids(detections: list) -> list:
        for det in detections:
            tid = det.get("track_id", 0)
            det["person_id"] = f"P{tid:03d}"
        return detections

    @staticmethod
    def _iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter + 1e-6)


# ====
#  FIREBASE MANAGER
# ====

class FirebaseManager:
    def __init__(self):
        self.db      = None
        self.enabled = False
        self._queue  = deque()
        self._lock   = threading.Lock()
        self._connect()
        if self.enabled:
            self._start_worker()

    def _connect(self):
        if not FIREBASE_AVAILABLE:
            return
        key_path = CONFIG["firebase_key"]
        if not Path(key_path).exists():
            print(f"[FIREBASE] Chave nao encontrada: {key_path}. Firebase desativado.")
            return
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(key_path)
                firebase_admin.initialize_app(cred)
            self.db      = firestore.client()
            self.enabled = True
            print("[FIREBASE] Conectado com sucesso.")
        except Exception as e:
            print(f"[FIREBASE] Erro ao conectar: {e}")

    def push(self, record: dict):
        if self.enabled:
            with self._lock:
                self._queue.append(record)

    def _start_worker(self):
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

    def _worker(self):
        col = CONFIG["firebase_col"]
        while True:
            if self._queue:
                with self._lock:
                    record = self._queue.popleft()
                try:
                    self.db.collection(col).add(record)
                except Exception as e:
                    print(f"[FIREBASE] Erro ao enviar: {e}")
            else:
                time.sleep(0.1)

    def get_employee(self, rfid: str) -> dict:
        if not self.enabled:
            return {}
        try:
            docs = (
                self.db.collection("funcionarios")
                .where("rfid", "==", rfid)
                .limit(1)
                .stream()
            )
            for doc in docs:
                return doc.to_dict()
        except Exception as e:
            print(f"[FIREBASE] Erro ao buscar funcionario: {e}")
        return {}

    def log_access(self, rfid: str, person_id: str,
                   liberado: bool, has_helmet: bool, has_vest: bool):
        record = {
            "rfid":       rfid,
            "person_id":  person_id,
            "liberado":   liberado,
            "has_helmet": has_helmet,
            "has_vest":   has_vest,
            "timestamp":  datetime.now().isoformat(),
        }
        self.push(record)


# ====
#  CONTROLE DE ACESSO
# ====

class AccessController:
    def __init__(self, firebase: FirebaseManager):
        self.firebase = firebase

    def evaluate(self, rfid: str, person_id: str,
                 has_helmet: bool, has_vest: bool) -> bool:
        ok = True
        if CONFIG["require_helmet"] and not has_helmet:
            ok = False
        if CONFIG["require_vest"] and not has_vest:
            ok = False

        if ok:
            print(f"[ACESSO] LIBERADO — RFID: {rfid} | ID: {person_id}")
        else:
            missing = []
            if not has_helmet: missing.append("CAPACETE")
            if not has_vest:   missing.append("COLETE")
            print(f"[ACESSO] BLOQUEADO — RFID: {rfid} | Faltando: {', '.join(missing)}")

        self.firebase.log_access(rfid, person_id, ok, has_helmet, has_vest)
        return ok


# ====
#  LOGGER LOCAL
# ====

class EPILogger:
    def __init__(self):
        CONFIG["log_dir"].mkdir(parents=True, exist_ok=True)
        self.session_start = datetime.now()
        self.records: list = []
        self._lock = threading.Lock()

        ts = self.session_start.strftime("%Y%m%d_%H%M%S")
        self.json_path = CONFIG["log_dir"] / f"epi_log_{ts}.json"
        self.txt_path  = CONFIG["log_dir"] / f"epi_summary_{ts}.txt"

        self._schedule_save()

    def add(self, person_id: str, rfid: str,
            has_helmet: bool, has_vest: bool,
            bbox: tuple, conf: float = 1.0) -> dict:
        record = {
            "timestamp":  datetime.now().isoformat(timespec="seconds"),
            "person_id":  person_id,
            "rfid":       rfid,
            "has_helmet": bool(has_helmet),
            "has_vest":   bool(has_vest),
            "violation":  not (has_helmet and has_vest),
            "bbox":       [int(v) for v in bbox],
            "conf":       float(conf),
        }
        with self._lock:
            self.records.append(record)
        return record

    def save_json(self):
        payload = {
            "session_start": self.session_start.isoformat(),
            "total":         len(self.records),
            "violations":    sum(1 for r in self.records if r["violation"]),
            "records":       self.records,
        }
        with self._lock:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    def save_txt(self):
        total      = len(self.records)
        violations = sum(1 for r in self.records if r["violation"])
        duration   = datetime.now() - self.session_start

        lines = [
            "=" * 65,
            "  RELATORIO FINAL - EPI MONITOR",
            "=" * 65,
            f"  Inicio   : {self.session_start.strftime('%d/%m/%Y %H:%M:%S')}",
            f"  Duracao  : {str(duration).split('.')[0]}",
            f"  Total    : {total}",
            f"  OK       : {total - violations}",
            f"  Violacoes: {violations}",
            "=" * 65, "",
        ]
        for r in self.records:
            status = "OK" if not r["violation"] else "VIOLACAO"
            h = "C" if r["has_helmet"] else "X"
            v = "C" if r["has_vest"]   else "X"
            lines.append(
                f"  [{r['timestamp']}] {r['person_id']:6s} RFID:{r['rfid']:12s} "
                f"Capacete:{h} Colete:{v} | {status}"
            )
        lines += ["", "=" * 65]

        with open(self.txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[LOG] Resumo salvo: {self.txt_path}")

    def _schedule_save(self):
        self.save_json()
        self._timer = threading.Timer(CONFIG["json_interval"], self._schedule_save)
        self._timer.daemon = True
        self._timer.start()

    def stop(self):
        if hasattr(self, "_timer"):
            self._timer.cancel()


# ====
#  DETECTOR EPI (GPU-OPTIMIZED)
# ====

class EPIDetector:
    def __init__(self):
        device = CONFIG["device"]
        print(f"[DETECTOR] Device: {device.upper()}")
        if device == "cuda":
            print(f"[DETECTOR] GPU: {torch.cuda.get_device_name(0)}")

        self.model = YOLO(CONFIG["model_path"])
        self.model.to(device)
        self.names = self.model.names

        self.person_ids = []
        self.helmet_ids = []
        self.vest_ids   = []

        for cid, name in self.names.items():
            nl = name.lower()
            if nl in PERSON_LABELS: self.person_ids.append(cid)
            if nl in HELMET_LABELS: self.helmet_ids.append(cid)
            if nl in VEST_LABELS:   self.vest_ids.append(cid)

        print(f"[DETECTOR] Pessoas  : {[self.names[c] for c in self.person_ids]}")
        print(f"[DETECTOR] Capacetes: {[self.names[c] for c in self.helmet_ids]}")
        print(f"[DETECTOR] Coletes  : {[self.names[c] for c in self.vest_ids]}")

    def detect(self, frame: np.ndarray) -> list:
        use_half = CONFIG["half"] and CONFIG["device"] == "cuda"

        results = self.model(
            frame,
            device=CONFIG["device"],
            imgsz=CONFIG["infer_size"],
            conf=CONFIG["conf"],
            iou=CONFIG["iou"],
            half=use_half,
            verbose=False,
        )[0]

        persons = []
        helmets = []
        vests   = []

        if results.boxes is None:
            return []

        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox = (x1, y1, x2, y2)

            if cls in self.person_ids:
                persons.append({"bbox": bbox, "conf": conf})
            elif cls in self.helmet_ids:
                helmets.append(bbox)
            elif cls in self.vest_ids:
                vests.append(bbox)

        detections = []
        for p in persons:
            px1, py1, px2, py2 = p["bbox"]
            ph = py2 - py1

            head_bbox  = (px1, py1, px2, py1 + int(ph * 0.25))
            torso_bbox = (px1, py1 + int(ph * 0.25), px2, py1 + int(ph * 0.75))

            has_helmet = any(
                self._iou(head_bbox, hb) > 0.15
                or self._overlap(head_bbox, hb) > 0.3
                for hb in helmets
            )
            has_vest = any(
                self._iou(torso_bbox, vb) > 0.15
                or self._overlap(torso_bbox, vb) > 0.3
                for vb in vests
            )

            detections.append({
                "bbox":       p["bbox"],
                "head_bbox":  head_bbox,
                "torso_bbox": torso_bbox,
                "has_helmet": has_helmet,
                "has_vest":   has_vest,
                "conf":       p["conf"],
            })

        return detections

    @staticmethod
    def _iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0: return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter + 1e-6)

    @staticmethod
    def _overlap(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter  = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / area_b


# ====
#  HUD RENDERER
# ====

class HUDRenderer:
    def __init__(self):
        self.log_buffer: deque = deque(maxlen=8)

    def render(self, frame: np.ndarray, detections: list,
               fps: float, access_status: dict = None) -> np.ndarray:
        out = frame.copy()
        for det in detections:
            self._draw_person(out, det)
        self._draw_hud(out, detections, fps)
        if access_status:
            self._draw_access_banner(out, access_status)
        return out

    def push_log(self, record: dict):
        ts   = record["timestamp"][-8:]
        pid  = record["person_id"]
        h    = "C" if record["has_helmet"] else "X"
        v    = "C" if record["has_vest"]   else "X"
        stat = "OK" if not record["violation"] else "VIOLACAO"
        self.log_buffer.append(f"[{ts}] {pid} H:{h} V:{v} | {stat}")

    def _draw_person(self, frame: np.ndarray, det: dict):
        x1, y1, x2, y2 = det["bbox"]
        has_helmet = det["has_helmet"]
        has_vest   = det["has_vest"]
        pid        = det.get("person_id", "?")

        if has_helmet and has_vest:
            color = CONFIG["color_ok"]
        elif has_helmet and not has_vest:
            color = CONFIG["color_warn"]
        else:
            color = CONFIG["color_danger"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        h_icon = "[C]" if has_helmet else "[X]"
        v_icon = "[V]" if has_vest   else "[X]"
        label  = f"{pid} H:{h_icon} V:{v_icon}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        if not has_helmet or not has_vest:
            missing = []
            if not has_helmet: missing.append("CAPACETE")
            if not has_vest:   missing.append("COLETE")
            msg = f"! SEM {' + '.join(missing)} !"
            (mw, mh), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cx = (x1 + x2) // 2
            tx = cx - mw // 2
            ty = y1 + mh + 30
            cv2.putText(frame, msg, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, CONFIG["color_danger"], 2)

    def _draw_hud(self, frame: np.ndarray, detections: list, fps: float):
        h, w    = frame.shape[:2]
        panel_w = 360
        panel_h = 30 + len(self.log_buffer) * 20 + 55
        mx, my  = 10, 10

        px1 = w - panel_w - mx
        py1 = h - panel_h - my
        px2 = w - mx
        py2 = h - my

        sub = frame[py1:py2, px1:px2]
        if sub.size:
            dark = np.zeros_like(sub)
            cv2.addWeighted(sub, 0.25, dark, 0.75, 0, sub)
            frame[py1:py2, px1:px2] = sub

        cv2.putText(frame, "LOG DE DETECCOES", (px1 + 8, py1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, CONFIG["color_yellow"], 1)

        for i, line in enumerate(self.log_buffer):
            c = CONFIG["color_danger"] if "VIOLACAO" in line else CONFIG["color_ok"]
            cv2.putText(frame, line, (px1 + 8, py1 + 38 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, c, 1)

        ok_count = sum(1 for d in detections if d["has_helmet"] and d["has_vest"])
        viol     = len(detections) - ok_count
        footer   = f"FPS:{fps:4.1f}  Pessoas:{len(detections)}  OK:{ok_count}  Viol:{viol}"
        cv2.putText(frame, footer, (px1 + 8, py2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, CONFIG["color_white"], 1)

        device_str = "GPU" if CONFIG["device"] == "cuda" else "CPU"
        header = f"EPI Monitor [{device_str}]  |  {datetime.now().strftime('%H:%M:%S')}"
        cv2.putText(frame, header, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, CONFIG["color_white"], 2)
        cv2.putText(frame, "Q = sair  |  R = simular RFID",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CONFIG["color_white"], 1)

    @staticmethod
    def _draw_access_banner(frame: np.ndarray, status: dict):
        liberado = status.get("liberado", False)
        rfid     = status.get("rfid", "")
        nome     = status.get("nome", "Desconhecido")
        elapsed  = time.time() - status.get("ts", time.time())

        if elapsed > 4.0:
            return

        h, w  = frame.shape[:2]
        msg   = "ACESSO LIBERADO" if liberado else "ACESSO BLOQUEADO"
        color = CONFIG["color_ok"] if liberado else CONFIG["color_danger"]

        overlay = frame.copy()
        cv2.rectangle(overlay, (w // 4, h // 3), (3 * w // 4, 2 * h // 3), color, -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
        cv2.putText(frame, msg, (w // 2 - tw // 2, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)

        sub_msg = f"RFID: {rfid}  |  {nome}"
        (sw, _), _ = cv2.getTextSize(sub_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(frame, sub_msg, (w // 2 - sw // 2, h // 2 + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


# ====
#  MONITOR PRINCIPAL
# ====

class EPIMonitor:
    def __init__(self):
        print(f"\n{'='*60}")
        print("  EPI Monitor — Iniciando sistema...")
        print(f"{'='*60}\n")

        self.detector    = EPIDetector()
        self.tracker     = PersonTracker()
        self.logger      = EPILogger()
        self.firebase    = FirebaseManager()
        self.access_ctrl = AccessController(self.firebase)
        self.hud         = HUDRenderer()

        self._last_log: dict       = {}
        self._access_status: dict  = None
        self._pending_rfid: str    = None
        self._rfid_lock            = threading.Lock()

    def simulate_rfid(self, rfid: str):
        with self._rfid_lock:
            self._pending_rfid = rfid

    def run(self):
        cap = cv2.VideoCapture(CONFIG["camera_index"])
        if not cap.isOpened():
            raise RuntimeError(
                f"Camera nao encontrada (indice {CONFIG['camera_index']}). "
                "Verifique se a webcam esta conectada."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CONFIG["frame_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["frame_height"])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        fps_buf   = deque(maxlen=30)
        prev_time = time.time()

        print(f"[SISTEMA] Device    : {CONFIG['device'].upper()}")
        print(f"[SISTEMA] Modelo    : {CONFIG['model_path']}")
        print(f"[SISTEMA] Camera    : indice {CONFIG['camera_index']}")
        print("[SISTEMA] Pressione Q para sair | R para simular RFID\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                # ── Detecção + Tracking ────
                raw_dets   = self.detector.detect(frame)
                detections = self.tracker.update(raw_dets)

                # ── Logging com cooldown ────
                now = time.time()
                for det in detections:
                    pid = det.get("person_id", "P000")
                    if now - self._last_log.get(pid, 0) >= CONFIG["log_cooldown"]:
                        record = self.logger.add(
                            person_id  = pid,
                            rfid       = self._pending_rfid or "",
                            has_helmet = det["has_helmet"],
                            has_vest   = det["has_vest"],
                            bbox       = det["bbox"],
                            conf       = det.get("conf", 1.0),
                        )
                        self.hud.push_log(record)
                        self.firebase.push(record)
                        self._last_log[pid] = now

                # ── RFID pendente → avalia acesso ────
                with self._rfid_lock:
                    rfid = self._pending_rfid
                    self._pending_rfid = None

                if rfid and detections:
                    det      = detections[0]
                    employee = self.firebase.get_employee(rfid) or {}
                    liberado = self.access_ctrl.evaluate(
                        rfid       = rfid,
                        person_id  = det.get("person_id", "P000"),
                        has_helmet = det["has_helmet"],
                        has_vest   = det["has_vest"],
                    )
                    self._access_status = {
                        "liberado": liberado,
                        "rfid":     rfid,
                        "nome":     employee.get("nome", "Desconhecido"),
                        "ts":       time.time(),
                    }

                # ── FPS ────
                fps_buf.append(1.0 / max(time.time() - prev_time, 1e-6))
                prev_time = time.time()
                fps = float(np.mean(fps_buf))

                # ── Renderização ────
                output = self.hud.render(frame, detections, fps, self._access_status)
                cv2.imshow("EPI Monitor", output)

                # ── Teclas ────
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\n[SISTEMA] Encerrando...")
                    break
                elif key == ord("r"):
                    rfid_input = input("Digite o RFID: ").strip()
                    if rfid_input:
                        self.simulate_rfid(rfid_input)

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.logger.stop()
            self.logger.save_json()
            self.logger.save_txt()
            print(f"[SISTEMA] Logs em: {CONFIG['log_dir'].resolve()}")
            print("[SISTEMA] Encerrado.")


# ====
#  ENTRY POINT
# ====

if __name__ == "__main__":
    monitor = EPIMonitor()
    monitor.run()
