import cv2
import  numpy as np
from os import listdir
from os.path import  isfile ,join

#the location where dataset is located
data_path="C:/Users/lenovo/PycharmProjects/Machine Learning/opencv/Face Recognition System/dataset/"

# to check the reqires face exists or not
only_files=[f for f in listdir(data_path) if isfile(join(data_path,f))]

Training_data , Labels= [],[]

for i ,files in enumerate(only_files):
    # image path + file name
    image_path=data_path+only_files[i]

    # reading of image
    images=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

    # training data as array
    Training_data.append(np.asarray(images,dtype=np.uint8))
    Labels.append(i)

# array of labels
Labels=np.asarray(Labels,dtype=np.uint32)

# creation of model
model=cv2.face.LBPHFaceRecognizer_create()

# trainig of model
model.train(np.asarray(Training_data),np.asarray(Labels))

# now prediction procedure
face_classifier=cv2.CascadeClassifier('C:/Users/lenovo/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')


def face_detector(img,size=0.5):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,2,5)

    if faces is():
        return  None

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        roi=img[y:y+h,x:x+w]
        roi=cv2.resize(roi,(400,400))

    return img,roi

cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()

    # it returns image and region of interest
    image,face=face_detector(frame)

    try:
         face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)

         # prediction of face
         result=model.predict(face)

         if result[1] < 500:

             #it tells the percentag of matching face
             confidence=int(100*(1-(result[1])/300))

             display_string=str(confidence)+'% confidence it is user'
         cv2.putText(image,display_string,(100,120),cv2.FONT_HERSHEY_SIMPLEX)

         if confidence <75:
             cv2.putText(image,"Unlocked",(250,450),cv2.FONT_HERSHEY_SIMPLEX)
             cv2.imshow('Face',image)
         else:
             cv2.putText(image,"Locked",(250,450),cv2.FONT_HERSHEY_SIMPLEX)
             cv2.imshow('Face',image)

    except:
        cv2.putText(image,"Face Not Found",(250,450),cv2.FONT_HERSHEY_SIMPLEX)
        cv2.imshow('Face',image)
        pass
    if cv2.waitKey(1)==13:
        break
cap.release()
cv2.destroyAllWindows()

