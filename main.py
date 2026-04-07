# main.py
import cv2
import time
import numpy as np
from datetime import datetime
from ultralytics import YOLO

from database import (
    register_session_start, register_session_end,
    register_violation, get_total_violations,
)
from validator import validate_ppe_color
from config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    MODEL_PATH, CONFIDENCE_THRESHOLD, IOU_THRESHOLD,
    COLOR_SAFE, COLOR_UNSAFE, COLOR_PERSON,
    VIOLATION_COOLDOWN_SECONDS, FONT_SCALE, FONT_THICKNESS, BOX_THICKNESS,
    VIOLATIONS_DB_PATH,
)

# face utils
try:
    from face_utils import load_face_db, identify_face, crop_face_from_bbox
    FACE_AVAILABLE = True
except Exception:
    FACE_AVAILABLE = False
    print("[WARN] face_utils nao disponivel. Identificacao por nome desativada.")

# tracking config
_SAME_PERSON_DIST_PX  = 120
_PERSON_SLOT_EXPIRE_S = 5.0
_REPORT_INTERVAL_S    = 30

# known persons order (fallback names if not recognized)
KNOWN_PERSONS_ORDER = [
    "Felipe Fraga",
    "Milton Rafael",
]

NEGATIVE_KEYWORDS = {"no", "sem", "without", "missing", "bare", "exposed", "unprotected", "head"}
PERSON_KEYWORDS = {"person", "people", "worker", "human"}


def load_model() -> YOLO:
    import os
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Modelo PPE nao encontrado localmente. Tentando baixar modelos especializados...")
        for model_id in [
            "keremberke/yolov8n-hard-hat-detection",
            "keremberke/yolov8s-hard-hat-detection",
        ]:
            try:
                model = YOLO(model_id)
                model.save(MODEL_PATH)
                print(f"[INFO] Modelo salvo em: {MODEL_PATH}")
                return model
            except Exception as e:
                print(f"[AVISO] Falha ao baixar {model_id}: {e}")
        print("[AVISO] Usando modelo YOLOv8 base (deteccao de pessoas apenas).")
        return YOLO("yolov8n.pt")
    print(f"[INFO] Carregando modelo: {MODEL_PATH}")
    return YOLO(MODEL_PATH)


def draw_label(frame: np.ndarray, text: str, x1: int, y1: int, color: tuple) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)
    y_label = max(y1 - 8, th + 8)
    cv2.rectangle(frame, (x1, y_label - th - 4), (x1 + tw + 4, y_label + 4), color, -1)
    cv2.putText(frame, text, (x1 + 2, y_label), cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, fps: float, total_violations: int,
             last_violation_ts: float, person_slots: dict) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (0 + w, 50), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    now_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    cv2.putText(frame, f"EPI Monitor | {now_str}", (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}  |  Pessoas: {len(person_slots)}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1, cv2.LINE_AA)
    viol_color = (0, 80, 255) if total_violations > 0 else (100, 255, 100)
    cv2.putText(frame, f"Violacoes: {total_violations}", (w - 260, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, viol_color, 1, cv2.LINE_AA)
    if last_violation_ts > 0:
        elapsed = int(time.time() - last_violation_ts)
        cv2.putText(frame, f"Ultima: {elapsed}s atras", (w - 260, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1, cv2.LINE_AA)

    # painel por pessoa (inferior direita)
    y_offset = h - 15
    for slot in reversed(list(person_slots.values())):
        name = slot.get("person_name", "Desconhecido")
        count = slot.get("violation_count", 0)
        color = (0, 80, 255) if count > 0 else (100, 255, 100)
        cv2.putText(frame, f"{name}: {count} viol", (w - 310, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        y_offset -= 18

    cv2.putText(frame, "Pressione Q para sair", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


def is_safe_class(class_name: str) -> bool | None:
    name_lower = class_name.lower().replace("-", " ").replace("_", " ")
    tokens = set(name_lower.split())
    if tokens & NEGATIVE_KEYWORDS:
        return False
    if tokens & PERSON_KEYWORDS:
        return None
    return True


def _centroid(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int]:
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def _resolve_person_name(slot_index: int) -> str:
    if slot_index < len(KNOWN_PERSONS_ORDER):
        return KNOWN_PERSONS_ORDER[slot_index]
    return f"Desconhecido {slot_index + 1}"


def _find_or_create_slot(slots: dict, cx: int, cy: int, bbox: tuple) -> str:
    best_id = None
    best_score = -1.0
    for slot_id, slot in slots.items():
        iou_score = _iou(bbox, slot.get("last_bbox", bbox))
        sx, sy = slot["centroid"]
        dist = ((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5
        dist_score = max(0.0, 1.0 - dist / _SAME_PERSON_DIST_PX)
        score = 0.6 * iou_score + 0.4 * dist_score
        if score > best_score:
            best_score = score
            best_id = slot_id

    if best_id is not None and best_score > 0.2:
        slots[best_id]["centroid"] = (cx, cy)
        slots[best_id]["last_bbox"] = bbox
        slots[best_id]["last_seen"] = time.time()
        return best_id

    slot_index = len(slots)
    new_id = f"slot_{slot_index}"
    slots[new_id] = {
        "centroid": (cx, cy),
        "last_bbox": bbox,
        "last_seen": time.time(),
        "last_violation_ts": 0.0,
        "violation_count": 0,
        "person_name": _resolve_person_name(slot_index),
    }
    print(f"[INFO] Nova pessoa detectada: {slots[new_id]['person_name']}")
    return new_id


def _expire_slots(slots: dict) -> None:
    now = time.time()
    expired = [k for k, v in slots.items() if now - v["last_seen"] > _PERSON_SLOT_EXPIRE_S]
    for k in expired:
        del slots[k]


def _print_partial_report(person_slots: dict) -> None:
    print(f"\n{'='*50}")
    print(f"[RELATORIO PARCIAL] {datetime.now().strftime('%H:%M:%S')}")
    if not person_slots:
        print("  Nenhuma pessoa detectada no periodo.")
    for slot in person_slots.values():
        name = slot.get("person_name", "Desconhecido")
        count = slot.get("violation_count", 0)
        status = "⚠" if count > 0 else "✓"
        print(f"  {status} | {name}: {count} violacao(oes)")
    print(f"{'='*50}\n")


def _print_final_report(person_slots: dict) -> None:
    print(f"\n{'='*50}")
    print(f"[RESUMO FINAL] {datetime.now().strftime('%H:%M:%S')}")
    for slot in person_slots.values():
        name = slot.get("person_name", "Desconhecido")
        count = slot.get("violation_count", 0)
        print(f"  {name} esta com {count} violacao(oes) no periodo ativo do programa.")
    print(f"{'='*50}\n")


def run_detection() -> None:
    model = load_model()
    model_classes = model.names if hasattr(model, "names") else {}

    face_db = {}
    if FACE_AVAILABLE:
        try:
            face_db = load_face_db()
        except Exception as e:
            print(f"[WARN] Nao foi possivel carregar face_db: {e}")
            face_db = {}

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERRO] Nao foi possivel abrir a camera indice {CAMERA_INDEX}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    session_id = register_session_start()
    print(f"[INFO] Sessao iniciada: {session_id}")
    print(f"[INFO] Pessoas cadastradas: {', '.join(KNOWN_PERSONS_ORDER)}")
    print(f"[INFO] Relatorio parcial a cada {_REPORT_INTERVAL_S}s")
    print("[INFO] Pressione Q para encerrar.\n")

    frame_count = 0
    fps = 0.0
    fps_timer = time.time()
    fps_frame_count = 0
    last_violation_ts: float = 0.0
    last_report_ts = time.time()
    total_violations = get_total_violations()
    person_slots: dict = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[AVISO] Frame nao capturado. Encerrando...")
            break

        frame_count += 1
        fps_frame_count += 1
        now = time.time()

        _expire_slots(person_slots)

        if now - last_report_ts >= _REPORT_INTERVAL_S:
            _print_partial_report(person_slots)
            last_report_ts = now

        results = model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                raw_name = model_classes.get(cls_id, str(cls_id))
                detections.append({
                    "cls_id": cls_id,
                    "conf": conf,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "raw_name": raw_name,
                    "safe_status": is_safe_class(raw_name),
                })

        slot_violations = {}

        for det in detections:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            raw_name = det["raw_name"]
            conf = det["conf"]
            safe_status = det["safe_status"]

            cx, cy = _centroid(x1, y1, x2, y2)
            slot_id = _find_or_create_slot(person_slots, cx, cy, (x1, y1, x2, y2))
            slot = person_slots[slot_id]
            name = slot.get("person_name", _resolve_person_name(int(slot_id.split("_")[1])))

            # tentar identificar rostro (se face_db disponivel)
            if FACE_AVAILABLE and face_db:
                try:
                    face_crop = crop_face_from_bbox(frame, (x1, y1, x2, y2))
                    if face_crop is not None:
                        person_name, dist = identify_face(face_crop, face_db)
                        if person_name:
                            slot["person_name"] = person_name
                            name = person_name
                except Exception:
                    pass

            if safe_status is True:
                color_valid, color_ratio, _ = validate_ppe_color(frame, raw_name, x1, y1, x2, y2)
                if color_valid:
                    color = COLOR_SAFE
                    label = f"{raw_name} {conf:.0%}"
                else:
                    color = COLOR_UNSAFE
                    label = f"! Roupa comum (sem {raw_name})"
                    slot_violations.setdefault(slot_id, []).append(f"Roupa comum no lugar de {raw_name}")

            elif safe_status is False:
                color = COLOR_UNSAFE
                label = f"! {raw_name} {conf:.0%}"
                slot_violations.setdefault(slot_id, []).append(raw_name)

            else:
                color = COLOR_PERSON
                label = f"{name} {conf:.0%}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
            draw_label(frame, label, x1, y1, color)

        # registra violacoes por slot com cooldown
        for slot_id, missing_items in slot_violations.items():
            slot = person_slots.get(slot_id)
            if slot is None:
                continue
            cooldown_ok = (now - slot["last_violation_ts"]) >= VIOLATION_COOLDOWN_SECONDS
            if cooldown_ok:
                name = slot.get("person_name", "Desconhecido")
                register_violation(
                    session_id, missing_items, frame_count,
                    person_name=name
                )
                total_violations += 1
                slot["violation_count"] += 1
                slot["last_violation_ts"] = now
                last_violation_ts = now
                print(f"[VIOLACAO] {datetime.now().strftime('%H:%M:%S')} | {name} | Faltando: {', '.join(missing_items)}")
                print(f"[INFO] {name} esta com {slot['violation_count']} violacao(oes) no periodo ativo")

        if now - fps_timer >= 1.0:
            fps = fps_frame_count / (now - fps_timer)
            fps_timer = now
            fps_frame_count = 0

        draw_hud(frame, fps, total_violations, last_violation_ts, person_slots)
        cv2.imshow("Monitor de EPI - Seguranca do Trabalho", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Encerrando sessao...")
            break

    cap.release()
    cv2.destroyAllWindows()
    register_session_end(session_id)
    _print_final_report(person_slots)
    print(f"[INFO] Sessao encerrada: {session_id}")
    print(f"[INFO] Verifique o log em: '{VIOLATIONS_DB_PATH}'")


if __name__ == "__main__":
    run_detection()