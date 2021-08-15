import cv2 as cv

eyeCascade = cv.CascadeClassifier("Eye Detect\\haarcascades\\haarcascade_eye.xml")

img_color = cv.imread("e3.jpg")
img_gray  = cv.cvtColor(img_color,cv.COLOR_BGR2GRAY)

eye = eyeCascade.detectMultiScale(img_gray,scaleFactor=1.01,minNeighbors = 50)

if len(eye)>0:
   for (x,y,w,h) in  eye:
      our_image_rect = cv.rectangle(img_color,(x,y),(x+w,y+h),(0,0,255),2)
   cv.imshow("eye detect",our_image_rect)


   


