# debug_faces.py
import os
import cv2
import sys

REF_DIR = "references"
OUT_DIR = "debug_crops"
FACE_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

def ensure(p): 
    if not os.path.exists(p): os.makedirs(p, exist_ok=True)

def main():
    print("PWD:", os.getcwd())
    if not os.path.isdir(REF_DIR):
        print(f"[ERRO] Pasta '{REF_DIR}' nao encontrada no diretorio atual.")
        print("Liste os arquivos com: python -c \"import os; print(os.getcwd()); print(os.listdir())\"")
        return

    detector = cv2.CascadeClassifier(FACE_CASCADE)
    ensure(OUT_DIR)

    persons = sorted([d for d in os.listdir(REF_DIR) if os.path.isdir(os.path.join(REF_DIR, d))])
    if not persons:
        print(f"[ERRO] Nao foram encontradas subpastas em '{REF_DIR}'. Estrutura esperada: references/<Nome>/*.jpg")
        return

    for person in persons:
        person_path = os.path.join(REF_DIR, person)
        out_person = os.path.join(OUT_DIR, person.replace(" ", "_"))
        ensure(out_person)
        files = sorted([f for f in os.listdir(person_path) if f.lower().endswith(('.jpg','.jpeg','.png'))])
        print(f"\n=== Person: '{person}'  => {len(files)} arquivos encontrados ===")
        if not files:
            print("  [WARN] Nenhuma imagem com extensão .jpg/.jpeg/.png encontrada nessa pasta.")
            continue

        for fname in files:
            fpath = os.path.join(person_path, fname)
            img = cv2.imread(fpath)
            if img is None:
                print(f"  [ERRO] Falha ao abrir imagem: {fpath}")
                continue
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30,30))
            if len(faces) > 0:
                # salva o maior rosto detectado
                x, y, ww, hh = max(faces, key=lambda r: r[2]*r[3])
                crop = img[y:y+hh, x:x+ww]
                outp = os.path.join(out_person, f"{os.path.splitext(fname)[0]}_face.jpg")
                cv2.imwrite(outp, crop)
                print(f"  [OK] {fname}: rosto detectado ({ww}x{hh}), salvo em {outp}")
            else:
                # salva crop heuristico (parte superior da imagem)
                top_h = max(30, int(h * 0.4))
                ny1 = 0
                ny2 = min(h, top_h)
                nx1 = int(w * 0.15)
                nx2 = max(nx1 + 10, int(w * 0.85))
                crop = img[ny1:ny2, nx1:nx2]
                outp = os.path.join(out_person, f"{os.path.splitext(fname)[0]}_heuristic.jpg")
                cv2.imwrite(outp, crop)
                print(f"  [WARN] {fname}: nenhum rosto detectado -> salvo crop heuristico em {outp} (size {crop.shape[1]}x{crop.shape[0]})")

    print("\nDebug concluido. Verifique a pasta 'debug_crops' para checar os crops gerados.")
    print("Se os crops heurísticos estiverem ruins, adicione imagens com rosto mais próximas / melhore iluminacao.")

if __name__ == '__main__':
    main()