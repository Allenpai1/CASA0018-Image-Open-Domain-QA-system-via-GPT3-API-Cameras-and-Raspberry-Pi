import cv2
import matplotlib.pyplot as plt
import keras_ocr
import time

#cv2.namedWindow("camera",1)

video = 'http://admin:admin@10.129.196.27:8081/video'  # use for IP camera collection

capture = cv2.VideoCapture(0)  # for iphone use put video instead

#set the width, height and the exposure time
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

time.sleep(2)
capture.set(cv2.CAP_PROP_EXPOSURE, -8.0)

# button to close window
while capture.isOpened():
    success, img = capture.read()  # read image, one frame by frame
    cv2.imshow("Normal camera",img)
    img = cv2.resize(img, (0,0), fx = 0.8, fy = 0.8)  # increase resolution by decrease frames size by 0.8
    frame = cv2.flip(img, flipCode=1)
    key = cv2.waitKey(10)
    if key == 113:  # 'esc' key pressed
        print('Quit the camera window')
        break
    if key == 32: # space key pressed, ascii table
        filename = 'frames.jpg'
        cv2.imwrite(filename, img)  # save image to current file

        
# close camera
capture.release()
#cv2.destroyAllWindows("camera")