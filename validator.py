import cv2
import numpy as np

# Faixas HSV para cores de EPI reais
# Formato: (H_min, H_max, S_min, S_max, V_min, V_max)
_EPI_COLOR_PROFILES = {
    # Coletes
    "colete_amarelo":    [(20,  35,  130, 255, 130, 255)],
    "colete_laranja":    [(5,   20,  130, 255, 130, 255)],
    "colete_verde_lima": [(35,  85,  130, 255, 130, 255)],

    # Capacetes
    "capacete_branco":   [(0,   180, 0,   55,  170, 255)],
    "capacete_amarelo":  [(20,  35,  90,  255, 130, 255)],
    "capacete_laranja":  [(5,   20,  90,  255, 130, 255)],
    "capacete_vermelho": [(0,   5,   90,  255, 90,  255), (170, 180, 90, 255, 90, 255)],
    "capacete_azul":     [(100, 130, 90,  255, 70,  255)],

    # Luvas (preto, cinza escuro, azul marinho, verde)
    "luvas_preto":       [(0,   180, 0,   60,  0,   80)],
    "luvas_cinza":       [(0,   180, 0,   50,  80,  160)],
    "luvas_azul":        [(100, 130, 60,  255, 40,  200)],
    "luvas_verde":       [(35,  85,  60,  255, 40,  200)],

    # Botinas (marrom, preto, bege)
    "botina_preta":      [(0,   180, 0,   60,  0,   70)],
    "botina_marrom":     [(5,   20,  60,  200, 30,  130)],
    "botina_bege":       [(15,  30,  20,  100, 140, 220)],
}

# Quais profiles se aplicam a qual classe detectada pelo modelo
_CLASS_PROFILES = {
    # Colete
    "vest":         ["colete_amarelo", "colete_laranja", "colete_verde_lima"],
    "safety vest":  ["colete_amarelo", "colete_laranja", "colete_verde_lima"],
    "colete":       ["colete_amarelo", "colete_laranja", "colete_verde_lima"],

    # Capacete
    "helmet":       ["capacete_branco", "capacete_amarelo", "capacete_laranja",
                     "capacete_vermelho", "capacete_azul"],
    "hardhat":      ["capacete_branco", "capacete_amarelo", "capacete_laranja",
                     "capacete_vermelho", "capacete_azul"],
    "capacete":     ["capacete_branco", "capacete_amarelo", "capacete_laranja",
                     "capacete_vermelho", "capacete_azul"],

    # Luvas
    "gloves":       ["luvas_preto", "luvas_cinza", "luvas_azul", "luvas_verde"],
    "luvas":        ["luvas_preto", "luvas_cinza", "luvas_azul", "luvas_verde"],

    # Botinas
    "boots":        ["botina_preta", "botina_marrom", "botina_bege"],
    "botina":       ["botina_preta", "botina_marrom", "botina_bege"],
    "safety boots": ["botina_preta", "botina_marrom", "botina_bege"],
}

# Percentual mínimo de pixels da cor EPI dentro da bbox para considerar válido
# Reduzido para 0.08 para tolerar iluminação adversa e ângulos diferentes
_MIN_COLOR_RATIO = 0.08


def _crop_roi(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _normalize_brightness(roi: np.ndarray) -> np.ndarray:
    """
    Aplica equalização de histograma no canal V (brilho) do HSV
    para reduzir impacto de iluminação adversa (sombra, contraluz).
    """
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    return hsv


def _color_ratio(hsv_roi: np.ndarray, profiles: list[str]) -> float:
    total_pixels = hsv_roi.shape[0] * hsv_roi.shape[1]
    if total_pixels == 0:
        return 0.0

    combined_mask = np.zeros((hsv_roi.shape[0], hsv_roi.shape[1]), dtype=np.uint8)

    for profile_name in profiles:
        ranges = _EPI_COLOR_PROFILES.get(profile_name, [])
        for (h_min, h_max, s_min, s_max, v_min, v_max) in ranges:
            lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
            upper = np.array([h_max, s_max, v_max], dtype=np.uint8)
            mask = cv2.inRange(hsv_roi, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

    matched = int(np.count_nonzero(combined_mask))
    return matched / total_pixels


def validate_ppe_color(
    frame: np.ndarray,
    class_name: str,
    x1: int, y1: int, x2: int, y2: int,
) -> tuple[bool, float, str]:
    """
    Valida se a região detectada tem cor compatível com EPI real.

    Retorna:
        (is_valid, color_ratio, reason)
        - is_valid:    True se passou na validação de cor
        - color_ratio: percentual de pixels com cor EPI (0.0 a 1.0)
        - reason:      descrição do resultado
    """
    name_lower = class_name.lower().replace("-", " ").replace("_", " ")

    profiles = None
    for key, profs in _CLASS_PROFILES.items():
        if key in name_lower:
            profiles = profs
            break

    # Classe sem perfil de cor definido → aceita sem validação
    if profiles is None:
        return True, 1.0, "classe sem validacao de cor"

    roi = _crop_roi(frame, x1, y1, x2, y2)
    if roi is None or roi.size == 0:
        return False, 0.0, "regiao invalida"

    # Tenta primeiro sem normalização
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    ratio = _color_ratio(hsv_roi, profiles)

    # Se falhou, tenta com equalização de brilho (tolerância a iluminação ruim)
    if ratio < _MIN_COLOR_RATIO:
        hsv_norm = _normalize_brightness(roi)
        ratio_norm = _color_ratio(hsv_norm, profiles)
        if ratio_norm >= _MIN_COLOR_RATIO:
            return True, ratio_norm, f"cor EPI confirmada com normalizacao ({ratio_norm:.0%})"
        return False, ratio_norm, f"cor nao compativel com EPI ({ratio_norm:.0%})"

    return True, ratio, f"cor EPI confirmada ({ratio:.0%})"