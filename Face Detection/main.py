
#open cv library
import cv2;
import os.path;

file_location = os.path.isfile("face.jpg")
#making a variable to store the data of all the faces 
trained_faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#mae a variable to show the image 
img = cv2.imread("C:\\Users\\dck\\Desktop\\Python AI Projects\\Face Detection\\download.jpg")

#convert the rgb image to grayscale image 
grayscale_img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

#comapring the image with the data from above and detecing the faces
face_detector_cordinates = trained_faces.detectMultiScale(grayscale_img)
print(face_detector_cordinates)

#create a rectangle around the face
for (x,y,w,h) in face_detector_cordinates:
    cv2.rectangle(img , (x , y) , (x+w , y+h) , (0,255,0) , 2)

#to show the image we have selected
cv2.imshow("Face detector" , img)

#variable that wait for any key to press and close the program
cv2.waitKey()


#code is completed
print("Code Is completed")