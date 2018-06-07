import numpy as np
import cv2
import argparse

#Function to obtain the center of the face
def centroid(x,y,w,h):
	center=[]
	center.append(x+w*0.5)
	center.append(y+h*0.5)
	print(center)
	return center

#Function to plot the center of the face
def center_plot(img,centroid):
	center=centroid(x,y,w,h)
	cv2.circle(img,(int(center[0]),int(center[1])),1,(0,0,255),-1)

#Importing the required classifiers here
face_cascade=cv2.CascadeClassifier('E:/OpenCV/Image-Processing/opencv/tree/master/data/haarcascades/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('E:/OpenCV/Image-Processing/opencv/tree/master/data/haarcascades/data/haarcascades/haarcascade_eye.xml')

#Image to be detected has been introduced here
ap=argparse.ArgumentParser("Enter the image")
ap.add_argument("-i","--Image",required=True,help="Add the face here")
args=vars(ap.parse_args())
img=cv2.imread(args["Image"])
 
#Resizing image so that it is of a managable size for viewing
img=cv2.resize(img,None,fx=0.5,fy=0.5)

cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#Conversion to grayscale
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#Faces detected have their co-orinates stored here
faces=face_cascade.detectMultiScale(img,1.3,5)

#Variable used to detect the number of faces, a counter basically
i=0
#Looping over coordinates, building squares around the faces
for(x,y,w,h) in faces:
	cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	roi_gray=gray[y:y+h,x:x+w]
	roi_color=img[y:y+h,x:x+w]
	eyes=eye_cascade.detectMultiScale(roi_gray)
	i+=1
	#print("No. of faces seen in image: {}\n".format(i))
	for(ex,ey,ew,eh) in eyes:
		cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
print("No. of faces seen in the image:{}\n".format(i))
#Facial Co-ordinates are displayed here
print("\n[INFO]Facial Co-ordinates:{}\n[INFO]Shape of Faces Matrix{}\n".format(faces,faces.shape))

#Calling the centroid function here
print(centroid(x,y,w,h))

#Function to plot the center of the face
center_plot(img,centroid)

#Image with the squares shown here
cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()