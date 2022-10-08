#!/usr/bin/env python2.7

import numpy as np
import math
import cv2 as cv
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
bridge = CvBridge()

def imgCallback(img):
    try:
        img = bridge.imgmsg_to_cv2(img, "bgr8")
    except CvBridgeError as e:
        print(e)
    else:
	cv.imshow('Image', img)

	#Pothole detection
	h = len(img)
	w = len(img[1])
	#print(int(math.ceil(0.2*h)+1))
	crop = img[int(math.ceil(0.2*h)+1):h,0:w,0:3]

	#To grayscale
	crop_grey = cv.cvtColor(crop,cv.COLOR_BGR2GRAY)

	#Adaptive threshold
	ret,thresh = cv.threshold(crop_grey,175,255,cv.THRESH_BINARY)

	#Morph open
	kernel_ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
	thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel_ellipse, iterations=10)
	cv.namedWindow('Morphed open', cv.WINDOW_NORMAL)
	cv.imshow('Morphed open',thresh)

  #Publish pothole-only mask 
  image_pub = rospy.Publisher("tocmYP/image", Image, queue_size=10)

  try:
    image_pub.publish(bridge.cv2_to_imgmsg(thresh, encoding="passthrough"))
  except CvBridgeError as e:
    print(e)
  cv.waitKey(1)

def main():

    #Subscribe to the camera feed topic
    rospy.init_node('camera_sub_pub')
    rospy.Subscriber("camera/image", Image, imgCallback)
    
    rospy.spin()


if __name__ == "__main__":
    main()
