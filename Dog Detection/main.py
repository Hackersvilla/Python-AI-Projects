

import cv2
from random import randrange
path_image = "C:\\Users\\dck\\Desktop\\Python AI Projects\\Dog Detection\\group_img.jpg"
data = cv2.CascadeClassifier("cat.xml")
img = cv2.imread(path_image)
grayscale_img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
cordintes_img = data.detectMultiScale(img)
for (x,y,w,h) in cordintes_img:
    cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 2)
cv2.imshow("Dog Image Classifier",img)
cv2.waitKey()