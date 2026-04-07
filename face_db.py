# face_db.py
import os
import json
import face_recognition

REF_DIR = "references"
OUT_FILE = "face_encodings.json"
MIN_IMAGES_PER_PERSON = 1  # recomendo >=2

def build_face_db(ref_dir=REF_DIR, out_file=OUT_FILE):
    db = {}
    if not os.path.isdir(ref_dir):
        print(f"[ERRO] Pasta de referencias nao existe: {ref_dir}")
        return

    for person_name in sorted(os.listdir(ref_dir)):
        person_path = os.path.join(ref_dir, person_name)
        if not os.path.isdir(person_path):
            continue
        encodings = []
        for fname in sorted(os.listdir(person_path)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(person_path, fname)
            try:
                img = face_recognition.load_image_file(img_path)
                faces = face_recognition.face_encodings(img)
                if not faces:
                    print(f"[WARN] Nenhum rosto detectado em {img_path}")
                    continue
                encodings.append(faces[0].tolist())
                print(f"[INFO] {person_name}: embedding gerado para {fname}")
            except Exception as e:
                print(f"[ERRO] Ao processar {img_path}: {e}")

        if len(encodings) >= MIN_IMAGES_PER_PERSON:
            db[person_name] = encodings
        else:
            print(f"[WARN] Ignorando {person_name}: menos que {MIN_IMAGES_PER_PERSON} imagens validas")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Banco de faces salvo em {out_file}. Pessoas: {list(db.keys())}")


if __name__ == "__main__":
    build_face_db()