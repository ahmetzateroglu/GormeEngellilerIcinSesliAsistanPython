#terminal pip install ultralytics
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import Tracker

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture("Video.mp4")

# Sınıfların listesini içeren bir dosyayı okuma
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
print(class_list)

count = 0


area = [(550, 450), (550, 50), (800, 50), (800, 450)]

# Nesneleri takip etmek için bir Tracker nesnesi oluşturma
tracker = Tracker()
#area_c = set()

while True:
    # Bir frame okuma
    ret, frame = cap.read()
    # Okuma başarısızsa döngüyü sonlandır
    if not ret:
        break
    # Her üç karede bir işlem yap
    count += 1
    if count % 3 != 0:
        continue

    # frame boyutunu ayarlama
    frame = cv2.resize(frame, (1020, 500))

    # YOLO modelini kullanarak karedeki nesneleri tespit et
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    list=[]
    for index, row in px.iterrows():
        #print(row)
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]

        # Eğer sınıf "bird" ise, listeye ekle
        if 'bird' in c:
            list.append([x1, y1, x2, y2, c])

    # Nesneleri takip et
    bbox_id = tracker.update(list)

    for bbox in bbox_id:
        x3, y3, x4, y4, obj_id, cl = bbox
        cx, cy = (x3 + x4) // 2, (y3 + y4) // 2

        # Eğer nesne belirli bir bölge içindeyse işlem yap
        results = cv2.pointPolygonTest(np.array(area, np.int32), ((cx, cy)), False)
        if results >= 0:
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
            cv2.putText(frame, str(c), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)
            #area_c.add(obj_id)
            if 'bird' in c:
                cv2.putText(frame, str(c), (50, 80), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 2)

    # Belirli bir bölgeyi gösteren çokgeni çiz
    cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 255, 0), 3)
    #count = len(area_c)

    # Sayacı ekrana yazdırma
    #cv2.putText(frame, str(count), (50, 140), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 2)

    # frame göster
    cv2.imshow("RGB", frame)

    # Eğer 'ESC' tuşuna basılırsa döngüden çık
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Video oynatmayı durdur ve pencereyi kapat
cap.release()
cv2.destroyAllWindows()
