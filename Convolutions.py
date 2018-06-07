from skimage.exposure import rescale_intensity
import numpy as np
from skimage.measure import compare_ssim
import argparse
import cv2
import imutils
import scipy as sp

#Convolution function, takes in the image and the kernel for processing, applies the required filters required
def convolve(image, kernel):
	(iH,iW)=image.shape[:2]
	(kH,kW)=kernel.shape[:2]
	pad=(kW-1)/2
	pad=int(pad)
	image=cv2.copyMakeBorder(image,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
	output=np.zeros((iH,iW),dtype="float32")

	for y in np.arange(pad,iH+pad):
		for x in np.arange(pad,iW+pad):
			roi=image[y-pad:y+pad+1,x-pad:x+pad+1]
			k=(roi*kernel).sum()
			output[y-pad,x-pad]=k
	output=rescale_intensity(output,in_range=(0,255))
	output=(output*255).astype("uint8")
	return output

def run():
	#Standard argument parser intro'd here, takes into consideration two images, that have to be compared
	ap=argparse.ArgumentParser()
	ap.add_argument("-i","--first_image",required=True,help="Add the path to the first image")
	ap.add_argument("-j","--second_image",required=True,help="Add the path to the second image")
	args=vars(ap.parse_args())


#Kernels are defined here

#Blurring kernels
	smallBlur=np.ones((7,7),dtype="float")*(1.0/(7.0*7.0))
	largeBlur=np.ones((21,21),dtype="float")*(1.0/(21*21))

#Sharpening Kernels
	sharpen=np.array((
	[0,-1,0],
	[-1,5,-1],
	[0,-1,0]),dtype="int")

#Laplacian Kerner
	laplacian=np.array((
	[0,1,0],
	[1,-4,1],
	[0,1,0]),dtype="int")

#Sobel Kernel in the X direction, detects the edges only in the X direction
	sobelX=np.array((
	[-1,0,1],
	[-2,0,2],
	[-1,0,1]),dtype="int")

#Sobel Kernel in the Y direction, detects the edges only in the Y direction
	sobelY=np.array((
	[-1,-2,-1],
	[0,0,0],
	[1,2,1]),dtype="int")

#Kernel bank, sends the image and the kernel to convolve function, to convolute it
	kernelBank=(
	("Small Bluring",smallBlur),
	("Large Bluring",largeBlur),
	("Sobel X",sobelX),
	("Sobel Y",sobelY),
	("Laplacian",laplacian),
	("Sharpen",sharpen))

#Adding both images for comparison here
	image_1=cv2.imread(args["first_image"])
	image_2=cv2.imread(args["second_image"])

#Resizing the image here
#image_1=cv2.resize(image_1,(300,300))
#image_2=cv2.resize(image_2,(300,300))
	(width,height,channels)=image_1.shape

	image_1=cv2.resize(image_1,(0,0),fx=.5,fy=.5)
	image_2=cv2.resize(image_2,(height,width))
	image_2=cv2.resize(image_2,(0,0),fx=0.5,fy=0.5)

#Converting image to grayscale here
	gray_1=cv2.cvtColor(image_1,cv2.COLOR_BGR2GRAY)
	gray_2=cv2.cvtColor(image_2,cv2.COLOR_BGR2GRAY)

#Displaying images here
	cv2.imshow("First Image",image_1)
	cv2.imshow("Second Image",image_2)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#Defining the scorer function here, provides a difference between each pixel
	(score,diff)=compare_ssim(gray_1,gray_2,full=True)
	diff=(diff*255).astype("uint8")
	print("Score:\n{:.2f}\n".format(score))

#Thresholder & contour detector goes in here
	thresh=cv2.threshold(diff,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	cnts=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts=cnts[0] if imutils.is_cv2() else cnts[1]

#Contour cycler
	for c in cnts:
		print("Co-ordinates of contour:\n{}\n".format(c))
		(x,y,w,h)=cv2.boundingRect(c)
		#print("Co-ordinates of contour:\n{}\n{}\n{}\n{}\n".format(x,y,w,h))
		print("Starting point X:{}\nStarting point Y:{}\nEnding point X:{}\nEnding point: Y{}\n".format(x,y,w,h))
		cv2.rectangle(image_1,(x,y),(x+w,y+h),(0,0,255),3)
		cv2.rectangle(image_2,(x,y),(x+w,y+h),(255,0,0),3)
	#cv2.putText(image_1,"Here:",(x,y-10),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1,255,cv2.LINE_AA)
	#cv2.putText(image_2,"Here:",(x,y-10),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1,255,cv2.LINE_AA)
	cv2.imshow("Image 1",image_1)
	cv2.imshow("Image 2",image_2)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.imshow("Difference",diff)
	cv2.imwrite("Difference.jpg",diff)
	cv2.imshow("Thresh",thresh)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#Cycles through all the kernels available in the bank, compares with the image

	for(kernelName,kernel) in kernelBank:
		print("[INFO] applying {} kernel".format(kernelName))
		convolveOutput=convolve(gray_1,kernel)
		cv2.imwrite(kernelName +".jpg", convolveOutput)
		opencvOutput=cv2.filter2D(gray_1,-1,kernel)
	#No need to call difference image here
		(score)=compare_ssim(convolveOutput,opencvOutput,full=True)
		print("Score for {} kernel is: {:.4f}\n".format(kernelName,score[0]))
	#cv2.imshow("original",gray_1)
		cv2.imshow("{}-convolve".format(kernelName),convolveOutput)
		cv2.imshow("{}-opencv".format(kernelName),opencvOutput)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

run()