import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

neighbors = 30

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(grayscale, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 250, 0), 5)

        the_face = frame[y:y + h, x:x + w]

        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        smile = smile_cascade.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=neighbors)

        if len(smile) > 0:
            cv2.putText(frame, 'smiling', (x, y + h + 40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN,
                        color=(255, 255, 255))

    cv2.imshow("smile detector", frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()