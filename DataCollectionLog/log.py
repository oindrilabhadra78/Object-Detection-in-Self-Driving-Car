import shutil
import cv2
import os
import csv
import time
import urllib
import serial

# Read the video from specified path
# cam = cv2.VideoCapture('rtsp://admin:abc123@192.168.43.89/live/ch00_0')
# cam = cv2.VideoCapture('http://192.168.137.253:8080/video')

# Webcam
cam = cv2.VideoCapture(0)
fps = cam.get(cv2.CAP_PROP_FPS)

# frame
framecount = 0

filename = 'car_data.csv'
dir = 'image_data'

speed = 0
steer = 0

if not os.path.isdir(dir):
    try:
        # creating a folder named data
        # if os.path.exists(dir):
        #    shutil.rmtree(dir)

        os.makedirs(dir)

        # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')

# reading from frame
ret, frame = cam.read()

# if video is left
while ret:
    print(framecount)
    if int(framecount % (fps / 10.0)) == 0:
        timestamp = time.time()
        name = './image_data/' + str(timestamp) + '.jpg'

        with open(filename, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([timestamp])

        # writing the extracted images
        cv2.imwrite(name, frame)

    # increasing counter so that it will
    # show how many frames are created
    framecount += 1

    # reading from frame
    ret, frame = cam.read()

# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()
