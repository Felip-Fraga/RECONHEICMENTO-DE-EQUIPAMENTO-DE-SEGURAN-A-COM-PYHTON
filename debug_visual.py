# debug_visual.py
import os, cv2
REF_DIR = "references"
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
for person in sorted(os.listdir(REF_DIR)):
    pdir = os.path.join(REF_DIR, person)
    if not os.path.isdir(pdir): continue
    for f in sorted(os.listdir(pdir)):
        if not f.lower().endswith(('.jpg','.jpeg','.png')): continue
        fp = os.path.join(pdir, f)
        img = cv2.imread(fp)
        if img is None:
            print("Erro abrir", fp); continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30,30))
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img, f"{person} - {f} - faces:{len(faces)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2)
        cv2.imshow("debug", img); k = cv2.waitKey(0)
        if k == 27: exit()
cv2.destroyAllWindows()
print("Finalizado")