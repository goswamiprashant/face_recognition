import cv2
import numpy as np

# library matched face features from the image
face_classifier=cv2.CascadeClassifier('C:/Users/lenovo/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

def face_extractor(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # it detects face from inage
    faces=face_classifier.detectMultiScale(gray,2,5)

    if faces is():
        return None

    for (x,y,w,h) in faces:
        # cropping out face from image
        cropped_face=img[y:y+h,x:x+w]
    return  cropped_face

# accessing of web cam
cap=cv2.VideoCapture(0)
count=0

while True:
    ret,frame=cap.read()

    # now face ertractor
    if face_extractor(frame) is not None:
        count +=1
       # resizing and converting the image into gray scale
        face=cv2.resize(face_extractor(frame),(400,400))
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)

        #location where we want to loacte our dataset
        file_name_path="C:/Users/lenovo/PycharmProjects/Machine Learning/opencv/Face Recognition System/dataset/user"+str(count)+".jpg"
        cv2.imwrite(file_name_path,face)

        # showing counting of image on umage
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper',face)
    else:
        print("Face Not Found")
        pass

    if cv2.waitKey(1)==13 or count==1000:
        break

cap.release()
cv2.destroyAllWindows()
print("Collection of samples completed")
#================================================================================================================
