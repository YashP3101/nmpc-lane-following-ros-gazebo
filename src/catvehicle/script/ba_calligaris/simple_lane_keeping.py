#!/usr/bin/env python3
# rosrun catvehicle simple_lane_keeping.py

import roslib
import cv2
import numpy
import scipy.signal
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from std_msgs.msg import Float64
from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot
from pyclothoids import Clothoid
import camera_transformation as ct
import sys
import os

rospy.init_node('simple_lane_keeping')

# Publisher cmd_vel
cmd_vel = rospy.Publisher('/catvehicle/cmd_vel', Twist, queue_size=5000)
move_cmd = Twist()

rate = rospy.Rate(10) # 10hz
img_width = 800 # Pixel
img_height = 800 # Pixel

distance = 0


def display_image(img):
    cv2.imshow('Kamera', img)
    cv2.waitKey(2)


def find_lane_start(img):
    # Linien links und rechts vom Fahrzeug finden
    peaks_r = scipy.signal.find_peaks(img[540:550, 470:550].sum(axis=0), prominence=(500, None))[0] + 470
    peaks_l = scipy.signal.find_peaks(img[540:550, 220:320].sum(axis=0), prominence=(500, None))[0] + 220
    
    lr = 525
    ll = 275
    
    if peaks_r.size > 0:
        lr = peaks_r[numpy.abs(peaks_r - lr).argmin()]
    if peaks_l.size > 0:
        ll = peaks_l[numpy.abs(peaks_l - ll).argmin()]
  
    return ll, lr


def move_vehicle(lateral_error):
    forward_velocity = 10
    steering_angle = -numpy.arctan(lateral_error / 10)

    move_cmd.linear.x = forward_velocity
    move_cmd.angular.z = steering_angle

    cmd_vel.publish(move_cmd)


def camera_callback(image_data):
    # Eingangsbild
    input_image = numpy.frombuffer(image_data.data, dtype=numpy.uint8).reshape(image_data.height, image_data.width, -1)

    # -> schwartz weiss bild
    image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # -> thresholding
    ret, threshold_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

    edge_image = cv2.GaussianBlur(image, (5, 5), 0)

    # -> canny edge detection
    edge_image = cv2.Canny(edge_image, 220, 255)

    # Kombiniertes Bild
    combined_image = threshold_image + edge_image
   
    # find lane starting points
    lane_left_start, lane_right_start = find_lane_start(combined_image)

    
    display_image(combined_image)
    lateral_error = (lane_left_start + lane_right_start - 800) / 16
    move_vehicle(lateral_error)
            
# Subscribter zur Kanera in der Gazevo Simulation
rospy.Subscriber("/catvehicle/camera_left/image_raw_left", Image, camera_callback)

if __name__ == '__main__':
    while not rospy.is_shutdown():
        rate.sleep()


