import cv2
from face_teain_use_keras import Model
from mySerial import Open_door, Close_door, Warning_door, UNwarning_door
import time
if __name__ == '__main__':
    model = Model()
    model.load_model(file_path='me.face.model.h5')
    color = (255, 255, 255)
    cap = cv2.VideoCapture(0)
    cascade_path = "D://Users\Administrator/anaconda3\envs/tf\Library\etc\haarcascades\haarcascade_frontalface_default.xml"
    NUm = 3
    while True:
        ret, frame = cap.read()
        if ret != True:
            continue
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cascade_path)
        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect
                if w < 100:
                    continue
                image = frame[y-10:y+h+10, x-10:x+w+10]
                faceID = model.face_predict(image)
                if faceID == 0:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), 2)
                    cv2.putText(frame, 'me', (x + 40, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if NUm != faceID:
                        UNwarning_door()
                        NUm = faceID
                        Open_door()

                elif faceID == 1:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), 2)
                    cv2.putText(frame, 'tw', (x + 40, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if NUm != faceID:
                        UNwarning_door()
                        NUm = faceID
                        Open_door()
                elif faceID == 2:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), 2)
                    cv2.putText(frame, 'cdd', (x + 40, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if NUm != faceID:
                        Warning_door()
                        NUm = faceID

                else:
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), 2)
                    cv2.putText(frame, 'who', (x + 40, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if NUm != faceID:
                        Warning_door()
                        NUm = faceID

        cv2.imshow('ccewa', frame)
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            exit(0)
    cap.release()
    cv2.destroyAllWindows()