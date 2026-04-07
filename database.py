import json
import os
from datetime import datetime

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from config import VIOLATIONS_DB_PATH
_DB_PATH = os.path.join(_BASE_DIR, VIOLATIONS_DB_PATH)


def _load_db() -> dict:
    if not os.path.exists(_DB_PATH):
        return {"total_violations": 0, "sessions": [], "records": []}
    with open(_DB_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {"total_violations": 0, "sessions": [], "records": []}


def _save_db(data: dict) -> None:
    with open(_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def register_session_start() -> str:
    db = _load_db()
    session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    db["sessions"].append({
        "session_id":            session_id,
        "started_at":            datetime.now().isoformat(),
        "ended_at":              None,
        "duration_seconds":      None,
        "violations_in_session": 0,
    })
    _save_db(db)
    return session_id


def register_session_end(session_id: str) -> None:
    db = _load_db()
    for session in db["sessions"]:
        if session["session_id"] == session_id:
            session["ended_at"] = datetime.now().isoformat()
            try:
                started = datetime.fromisoformat(session["started_at"])
                ended   = datetime.fromisoformat(session["ended_at"])
                session["duration_seconds"] = int((ended - started).total_seconds())
            except Exception:
                session["duration_seconds"] = None
            break
    _save_db(db)


def register_violation(
    session_id:    str,
    missing_items: list[str],
    frame_number:  int,
    person_name:   str | None = None,
    track_id:      int | None = None,
) -> None:
    """
    Registra uma violação de EPI.

    Parâmetros:
        session_id:    ID da sessão ativa
        missing_items: lista de EPIs ausentes
        frame_number:  frame onde ocorreu a violação
        person_name:   nome da pessoa identificada (ex: 'Felipe Fraga')
        track_id:      ID da track ByteTrack (opcional)
    """
    db = _load_db()
    record = {
        "id":               db["total_violations"] + 1,
        "session_id":       session_id,
        "timestamp":        datetime.now().isoformat(),
        "frame_number":     frame_number,
        "track_id":         track_id,
        "person_name":      person_name or "Desconhecido",
        "missing_equipment": missing_items,
        "total_missing":    len(missing_items),
        "status":           "violation",
    }
    db["records"].append(record)
    db["total_violations"] += 1
    for session in db["sessions"]:
        if session["session_id"] == session_id:
            session["violations_in_session"] += 1
            break
    _save_db(db)


def get_total_violations() -> int:
    db = _load_db()
    return db.get("total_violations", 0)


def get_violations_by_person(person_name: str) -> list[dict]:
    """Retorna todas as violações registradas para uma pessoa específica."""
    db = _load_db()
    return [r for r in db["records"]
            if r.get("person_name", "").lower() == person_name.lower()]


def get_session_summary(session_id: str) -> dict | None:
    """Retorna resumo da sessão com contagem de violações por pessoa."""
    db = _load_db()
    session = next(
        (s for s in db["sessions"] if s["session_id"] == session_id), None)
    if session is None:
        return None

    session_records = [r for r in db["records"]
                       if r["session_id"] == session_id]

    by_person: dict[str, dict] = {}
    for rec in session_records:
        name = rec.get("person_name", "Desconhecido")
        if name not in by_person:
            by_person[name] = {
                "total_violations":       0,
                "missing_equipment_counts": {},
            }
        by_person[name]["total_violations"] += 1
        for item in rec.get("missing_equipment", []):
            counts = by_person[name]["missing_equipment_counts"]
            counts[item] = counts.get(item, 0) + 1

    return {
        "session_id":          session_id,
        "started_at":          session.get("started_at"),
        "ended_at":            session.get("ended_at"),
        "duration_seconds":    session.get("duration_seconds"),
        "total_violations":    session.get("violations_in_session", 0),
        "violations_by_person": by_person,
    }