################################################### FACE DETCHION ########################################################


# open cv library
import cv2
from random import randrange

path = "/Face Detection Using Image\\download.jpg"
d = "haarcascade_frontalface_default.xml"
# first we need to import our data
data = cv2.CascadeClassifier(d)
# second to get the image which we want to use
img = cv2.imread(path)
# converting the image to grayscale form
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# to get the coordinate of the image
cordinate_img = data.detectMultiScale(img)
for (x, y, w, h) in cordinate_img:
    cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 2)
# we will show the image
cv2.imshow("Face Detector", img)
print("Faces Detected ")
# finally
cv2.waitKey()
