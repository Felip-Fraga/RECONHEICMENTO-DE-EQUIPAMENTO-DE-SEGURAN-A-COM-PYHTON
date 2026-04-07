# lbph_train.py
import os
import json
import cv2
import numpy as np

REF_DIR = "references"
MODEL_OUT = "lbph_model.yml"
LABEL_MAP_OUT = "lbph_label_map.json"
FACE_SIZE = (160, 160)  # tamanho de entrada (w,h)
MIN_IMAGES_PER_PERSON = 1  # recomendo >=2

def collect_faces_from_refs(ref_dir=REF_DIR):
    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(haar_path)
    images = []
    labels = []
    label_map = {}
    label_idx = 0

    if not os.path.isdir(ref_dir):
        raise FileNotFoundError(f"Pasta de referencias nao encontrada: {ref_dir}")

    for person_name in sorted(os.listdir(ref_dir)):
        person_path = os.path.join(ref_dir, person_name)
        if not os.path.isdir(person_path):
            continue
        person_images = []
        for fname in sorted(os.listdir(person_path)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(person_path, fname)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARN] Nao foi possivel ler: {img_path}")
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            if len(faces) == 0:
                print(f"[WARN] Nenhum rosto detectado em {img_path} — ignore se for imagem de corpo inteiro.")
                # tentar usar região central superior (heurística)
                h, w = gray.shape
                top_h = max(30, int(h * 0.4))
                crop = gray[0:top_h, int(w*0.25):int(w*0.75)]
                try:
                    face_resized = cv2.resize(crop, FACE_SIZE)
                    person_images.append(face_resized)
                    print(f"[INFO] Usando crop heurístico para {img_path}")
                except Exception:
                    continue
            else:
                # use a maior detecção
                face = max(faces, key=lambda r: r[2]*r[3])
                x, y, w_f, h_f = face
                face_img = gray[y:y+h_f, x:x+w_f]
                face_resized = cv2.resize(face_img, FACE_SIZE)
                person_images.append(face_resized)
                print(f"[INFO] {person_name}: rosto extraido de {fname}")

        if len(person_images) >= MIN_IMAGES_PER_PERSON:
            label_map[str(label_idx)] = person_name
            images.extend(person_images)
            labels.extend([label_idx] * len(person_images))
            label_idx += 1
        else:
            print(f"[WARN] Pessoa '{person_name}' tem menos que {MIN_IMAGES_PER_PERSON} imagens validas — ignorada.")

    if not images:
        raise RuntimeError("Nenhuma imagem de rosto coletada. Verifique a pasta 'references/'.")
    return np.array(images), np.array(labels), label_map

def train_and_save(out_model=MODEL_OUT, out_label_map=LABEL_MAP_OUT):
    images, labels, label_map = collect_faces_from_refs()
    # Criar recognizer LBPH (requer opencv-contrib)
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except Exception as e:
        raise RuntimeError("cv2.face nao disponivel. Instale opencv-contrib-python.") from e

    recognizer.train(list(images), list(labels))
    recognizer.save(out_model)
    with open(out_label_map, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Treino concluido. Modelo salvo em: {out_model}. Label map salvo em: {out_label_map}")
    print(f"[INFO] Pessoas treinadas: {list(label_map.values())}")

if __name__ == "__main__":
    train_and_save()