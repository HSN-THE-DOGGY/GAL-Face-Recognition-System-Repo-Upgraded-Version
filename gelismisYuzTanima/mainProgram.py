import cv2
import face_recognition

# Tanınacak kişilerin yüzlerini yükleme
gorsel1 = face_recognition.load_image_file("jeffBezos.jpg")
gorsel1_tanimlama = face_recognition.face_encodings(gorsel1)[0]

gorsel2 = face_recognition.load_image_file("markZuckerberg.jpg")
gorsel2_tanimlama = face_recognition.face_encodings(gorsel2)[0]

tanimlanmis_yuzler = [
    gorsel1_tanimlama,
    gorsel2_tanimlama
]

# Video kaydı başlatma.
video_capture = cv2.VideoCapture(0)

# Yüzlerin konumlarını bir kez bulma
frame_count = 0
check_interval = 5  # Her 5 karede bir kontrol et
while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # Çerçeveyi küçült

    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    if face_encodings:
        break

    frame_count += 1

# Daha sonra yüz kontrolü yapma işlemi
while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # Çerçeveyi küçült

    # Yüzleri kodlayın
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(tanimlanmis_yuzler, face_encoding)

        if True in matches:
            first_match_index = matches.index(True)
            if first_match_index == 0:
                name = "Jeff Bezos"
            elif first_match_index == 1:
                name = "Mark Zuckerberg"

            # Yüzün etrafına bir kutu ve isim çizin (küçük çerçeve boyutlarına göre ayarla)
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Dikdörtgen çizimi
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

            # Şeffaf çizgi çizime.
            alpha = 0.3
            overlay = frame.copy()
            cv2.rectangle(overlay, (left, top), (right, bottom), (0, 0, 255), -1)  # Renk ayarı burada yapılır
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # İsim yazısı
            cv2.putText(frame, name, (left + 4, bottom - 2), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

            # Çapraz çizgi çizime.
            cv2.line(frame, (left, top), (right, bottom), (255, 255, 255), 2)
            cv2.line(frame, (left, bottom), (right, top), (255, 255, 255), 2)

    # Sonuçları gösterme.
    cv2.imshow('Video', frame)

    # 'q' tuşu ile programı kapatma.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
