#terminal pip install ultralytics
# pip install gTTS
# pip install pygame
# pip install pyttsx3

from gtts import gTTS
from pygame import mixer
import time
import pyttsx3
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import Tracker


def seslendir(metin):
    # Motoru başlat
    motor = pyttsx3.init()
    # Türkçe sesi bul
    for ses in motor.getProperty('voices'):
        if "turkish" in ses.languages:
            motor.setProperty('voice', ses.id)
            break
    # Metni ses olarak çal
    motor.say(metin)
    # Metnin tamamını seslendirmek için bekleme
    motor.runAndWait()


# YOLO modelini yükleyelim
model = YOLO('yolov8s.pt') #model = YOLO('yolov8n.pt')

# Video dosyasını açalım
cap = cv2.VideoCapture(0)

# Sınıfların listesini içeren bir dosyayı okuyalım
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
print(class_list)
count = 0

# İzlenen bölgeyi belirlemek için bir alan tanımlayalım
area=[(350,450),(350,50),(650,50),(650,450)]

# Nesneleri takip etmek için bir Tracker nesnesi oluşturalım
tracker = Tracker()
area_c = set()



kullanilmisid = 9999
baslangiczamani= time.time() - 5 # ilk açıldığında geçen zaman 5 den fazla olsun diye normalde alttada 5 ama aşağıda ifden önce tanımladım diye burda da değişmem gerek
gecenzaman = 6

while True:
    # Bir kareyi okuyalım
    ret, frame = cap.read()
    # Okuma başarısızsa döngüyü sonlandır
    if not ret:
        break
    # Her üç karede bir işlem yapalım
    count += 1
    if count % 3 != 0:
        continue

    # Kare boyutunu ayarlayalım
    frame = cv2.resize(frame, (1020, 500))

    # YOLO modelini kullanarak karedeki nesneleri tespit edelim
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

        # Eğer sınıf "bird" ise, listeye ekleyelim
        if 'person' in c:
            list.append([x1, y1, x2, y2, c])
            #print("Deneme1")  # Ekranın herhangi bir yerindeyse yazdırıyor

    # Nesneleri takip edelim
    bbox_id = tracker.update(list)

    for bbox in bbox_id:
        x3, y3, x4, y4, obj_id, cl = bbox
        cx, cy = (x3 + x4) // 2, (y3 + y4) // 2

        # Eğer nesne belirli bir bölge içindeyse işlem yapalım
        results = cv2.pointPolygonTest(np.array(area, np.int32), ((cx, cy)), False)
        if results >= 0:
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
            cv2.putText(frame, str(obj_id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)
            area_c.add(obj_id)
            if 'person' in c:
                cv2.putText(frame, str(c), (50, 80), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 2)
                #print("Deneme2")  # Tracking alanındaysa yazdırıyor

                gecenzaman = time.time() - baslangiczamani

                if kullanilmisid != obj_id and gecenzaman >= 5:
                    kullanilmisid = obj_id
                    baslangiczamani = time.time()
                    print(obj_id)
                    cv2.putText(frame, str(obj_id), (50, 140), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 2)
                    seslendir("İnsan")


    """ tts = gTTS("Bir insan algılandı", lang="tr")
                      tts.save("deneme.mp3")
                      mixer.init()
                      mixer.music.load("deneme.mp3")
                      mixer.music.play()
                      while mixer.music.get_busy():
                          time.sleep(1) """







    # Belirli bir bölgeyi gösteren çokgeni çizelim
    cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 255, 0), 3)
    count = len(area_c)

    # Sayacı ekrana yazdıralım
    #cv2.putText(frame, str(count), (50, 140), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 2)


    # Kareyi gösterelim
    cv2.imshow("RGB", frame)

    # Eğer 'ESC' tuşuna basılırsa döngüden çık
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Video oynatmayı ve pencereyi kapatmayı bitirelim
cap.release()
cv2.destroyAllWindows()
