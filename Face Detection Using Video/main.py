
import cv2
from random import randrange

# to record a video
# Here 0 is to record from our own webcam
video_rec = cv2.VideoCapture(0)

d = "haarcascade_frontalface_default.xml"
#first we need to import our data
data = cv2.CascadeClassifier(d)

#iretate ove frame
#every frame to another
while True :

    #read the current frame
    sucess_frame , frame = video_rec.read()

    #convert to grayscale
    grayscale_img = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)


    # recording frame
    frame_corndinate = data.detectMultiScale(grayscale_img)

    #making rectangle around face
    for (x, y, w, h) in frame_corndinate:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 2)

    # showing the video
    cv2.imshow("Face Detection Through Video ", frame)

    #final line
    cv2.waitKey(1)