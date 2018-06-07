#Importing the neccessary libraries here
import numpy as np
import cv2

#Here, the necessary camera is accessed. In this case the built-in webcam is positioned at '0'.
#This is where the difference between this and the older program appears.
cap=cv2.VideoCapture(0)

#CascadeClassifiers are already trained on faces and can detect a number of features on faces.
#These are in-built in OpenCV and can be trained further if required
face_cascade=cv2.CascadeClassifier('E:/OpenCV/Image-Processing/opencv/tree/master/data/haarcascades/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('E:/OpenCV/Image-Processing/opencv/tree/master/data/haarcascades/data/haarcascades/haarcascade_eye.xml')

#Selecting a font for the text to be overlaid on the image later. Others are available
font=cv2.FONT_HERSHEY_SIMPLEX

#while Video_Capture() reads true, this loop will continue to execute
while(True):
	#Read extracts an 'image' and 'ret'
	#If the image has been returned, the boolean expression 'True' will be returned.
	ret,img=cap.read()

	#The image is then converted to gray, for further processing.
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#From this gray image, the faces are detected. The image is scaled for easier processing.
	faces=face_cascade.detectMultiScale(img,1.3,5)
	i=0
	for(x,y,w,h) in faces:
		cv2.putText(img,'Face: '+str(i+1),(x,y-10),font,1,(255,0,0),2,cv2.LINE_AA)
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray=gray[y:y+h,x:x+h]
		roi_color=img[y:y+h,x:x+h]
		eyes=eye_cascade.detectMultiScale(roi_gray)
		i+=1
		for(ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	cv2.putText(img,'Number of faces : '+str(i),(0,50),font,1,(255,255,255),2,cv2.LINE_AA)
	cv2.imshow('OpenCV Window',img)

	if cv2.waitKey(1)&0xFF==ord('q'):
		break
cap.release()
cv2.destroyAllWindows()