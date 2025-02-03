import cv2
import numpy as np

print(cv2.__version__)

#create a numpy array with zeros to use as a blank image
image = np.zeros((512, 512, 3), np.uint8)
#image = cv.imread("image5.jpg")

# draw a green line on the image
cv2.line(image, (0,0), (511, 511), (0,255,0), 5)

#draw a red rectangle on the image
cv2.rectangle(image, (384, 0), (510, 128), (0,0,255), 3)

#draw a blue circle on the image
cv2.circle(image, (447, 63), 63, (255, 0,0), -1)

#draw an ellipse
cv2.ellipse(image,(256,256),(100,50),0,0,180,255,-1)

#draw a polygon
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(image,[pts],True,(0,255,255))

#add text
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image,'Works',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)

#display the image
cv2.imshow("Image", image)

# wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()