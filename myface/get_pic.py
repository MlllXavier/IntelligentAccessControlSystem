import cv2

path_name = "me"
catch_pic_num = 100
num = 0



def GetPictures():
    cap = cv2.VideoCapture(0)
    classfier = cv2.CascadeClassifier("D://Users\Administrator/anaconda3\envs/tf\Library\etc\haarcascades\haarcascade_frontalface_default.xml")
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(64, 64))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect
                if w < 200:
                    continue
                global num
                img_name = "%s/%d.jpg"%(path_name, num)
                image = frame[y - 10: y + h + 20, x - 10: x + w + 20]
                cv2.imwrite(img_name, image)
                num += 1
                if num == catch_pic_num:
                    break
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 20, y + h + 20), (255, 0, 0), 2)
                cv2.putText(frame, "num: %d"%num, (x+20, y+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
            cv2.imshow("me", frame)
            if num >= catch_pic_num:
                break
            key = cv2.waitKey(30) & 0xff
            if key == 27:
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    GetPictures()