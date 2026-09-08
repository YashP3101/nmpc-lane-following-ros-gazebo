#!/usr/bin/env python3
# rosrun catvehicle lane_detection_kp.py

import roslib
import cv2
import numpy
import scipy.signal
import scipy.optimize as spo
import rospy
from sensor_msgs.msg import Image  
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray
from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot
from pyclothoids import Clothoid
import camera_transformation as ct
import sys
import os

rospy.init_node('lane_detection_node')

# Publisher cmd_vel Steuerung Fahrzeug
cmd_vel = rospy.Publisher('/catvehicle/cmd_vel', Twist, queue_size=1)
move_cmd = Twist()
# Publisher für Kruemmung
pub_co = rospy.Publisher('/catvehicle/curvature_opposing', Float64, queue_size=10)
pub_cl = rospy.Publisher('/catvehicle/curvature_left', Float64, queue_size=10)
pub_cr = rospy.Publisher('/catvehicle/curvature_right', Float64, queue_size=10)

rate = rospy.Rate(10) # 10hz updaterate

img_width = 800
img_height = 800    # kameraaufloesung
cc = 1.3962634      # bildebene groesse
# Neues Kameraobjekt
Cam = ct.Camera(img_width, img_height, 0, 1.75, 0.75, cc, cc, 50, 50)

bridge = CvBridge()

# bild darstellen
def display_image(img):
    cv2.imshow('Kamera', img)
    cv2.waitKey(2)

# anfang der fahrbahnmarkierungen finden
def find_lane_start(img):
    # das bild wir in 3 teilbereiche ingeteilt, in denen der anfang der markierungen gesucht wird
    box_r = img[540:550, 450:550].sum(axis=0)
    box_l = img[540:550, 220:320].sum(axis=0)
    box_o = img[540:550, 0:100].sum(axis=0)
    # der schwerpunkt in dem bereich ist der anfang der markierungen
    a = numpy.arange(0, 100)
    weight_r = (box_r * a).sum() / box_r.sum(axis=0) + 450
    weight_l = (box_l * a).sum() / box_o.sum(axis=0) + 220
    weight_o = (box_o * a).sum() / box_l.sum(axis=0)

    return(round(weight_o), round(weight_l), round(weight_r), 545)

# sliding-window-algorithmus
def sliding_windows(img, lstx, starty, box_height, box_width):
    # anfangsparameter
    pos_x = lstx
    pos_y = starty
    a = numpy.arange(0, box_width * 2)
    lane_x = []
    lane_y = []

    # erste box
    box = img[pos_y - box_height:pos_y, pos_x - box_width:pos_x + box_width].sum(axis=0)
    # gewicht der ersten box ist diff unterschiedlich von vom startpunkt
    if box.sum() > 500:
        weight = (box * a).sum() / box.sum(axis=0)
        diff = round(weight - box_width)
    else:   
        diff = 0        
    
    a = numpy.arange(0, box_width * 2)
    # position der zweiten box = gewicht der ersten
    # um box height hoeher
    pos_x += diff
    pos_y -= box_height
    lane_x.append(pos_x)
    lane_y.append(800 - pos_y)

    box = img[pos_y - box_height:pos_y, pos_x - box_width:pos_x + box_width].sum(axis=0)
    
    while box.sum() > 500:       
        # schwerpunkt der zweiten box
        weight = (box * a).sum() / box.sum(axis=0)
        diff = round(weight - box_width)
        
        if pos_y > 470:
            box_height = 2
            box_width = 5
            a = numpy.arange(0, box_width * 2)

        # zweite box neu centrieren
        pos_x += diff
        box = img[pos_y - box_height:pos_y, pos_x - box_width:pos_x + box_width].sum(axis=0)
        
        # neu zentrierte box zur lite hinzufuegen
        lane_x.append(pos_x)
        lane_y.append(800 - pos_y)

        # vorgeschlagene position fuer neue box
        pos_x += (pos_x - lane_x[-2])
        pos_y -= box_height

        # zweite box bei pos_x und y vorlaufig platzieren
        box = img[pos_y - box_height:pos_y, pos_x - box_width:pos_x + box_width].sum(axis=0)

    lane_x = numpy.array(lane_x)
    lane_y = numpy.array(lane_y)
    
    return lane_x, lane_y

# mittelline verbessern
def interpolate_mid_lane(lx, ly, rx, ry, ox, oy):
    # rechte fahrspur und entgegengesetzte fahrspur auf gleicher länge abschneiden
    diff = len(oy) - len(ry)
    if diff > 0:
        oy = oy[0:-diff-1]
        ox = ox[0:-diff-1]
    elif diff < 0:
        ry = ry[0:diff-1]
        rx = rx[0:diff-1]
    # wenn die mittellinie (ly) kürzer ist, als die anderen Markierungen
    diff2 = len(ry) - len(ly)
    # dann die fehlenden punkte der ml mit dem durchschnittswert der anderen beiden fahrspuren auffüllen
    if diff2 > 0:
        ly = numpy.append(ly, ry[-diff2:])
        lx = numpy.append(lx, (ox[-diff2:] + rx[-diff2:])/2)

    return lx, ly, rx, ry, ox, oy

# least squares fehlerfunktion 
def lsq_error(param, lx, ly):
    ast, aend, ex = param
    lx = numpy.array(lx)
    ly = numpy.array(ly)
    # klothoide anhand der eingangsparameter erstellen
    clothoid0 = Clothoid.G1Hermite(lx[0], ly[0], ast, ex, ly[-1], aend)
    # x werte der klothoide bestimmen
    cl = [clothoid0.X((i - ly[0]) * (clothoid0.length / (ly[-1] - ly[0]))) for i in ly]

    # least squares distanz zwischen der klothoide und messwerten
    diff = numpy.power(cl - lx, 2)
    score = diff.sum()

    return score

# klothoide an daten anpassen
def fit_clothoid(lx, ly):
    # start und endwinkel der farbahnmarkierung abschätzen
    dy = ly[-1] - ly[-5]
    dx = lx[-5] - lx[-1]
    ang_end_guess = numpy.arctan(dx / dy) + numpy.pi / 2
    dy = ly[5] - ly[0]
    dx = lx[0] - lx[5]
    ang_st_guess = numpy.arctan(dx / dy) / 2 + numpy.pi / 2
    # parameter die angepasst werden sollen: startwinkel, endwinkel, endpunkt
    parameters = numpy.array([ang_st_guess, ang_end_guess, lx[-1]])
    # die lsq_error funkton wird minimiert
    result = spo.minimize(lsq_error, parameters, args=(lx, ly), tol = 0.05, options={'maxiter': 20})
    # parameter der angepassten klothoide
    ang_start, ang_end, x_end = result.x
    clothoid0 = Clothoid.G1Hermite(lx[0], ly[0], ang_start, x_end, ly[-1], ang_end)

    # kruemmung der klothoide bestimmen
    curvature = []
    for s in ly:
        curvature.append(clothoid0.ThetaD(s))
    # gibt kruemmung und x und y koordinaten der klothoide zurueck
    return numpy.array(curvature), clothoid0.SampleXY(ly.size)[0], clothoid0.SampleXY(ly.size)[1]

# erkannte fahrbahnmarkierungen aufs eingangsbild zeichnen    
def draw_lines_on_input(img, lx, ly):
    for e in range(len(ly) - 1):
        img = cv2.line(img, (lx[e], ly[e]), (lx[e+1], ly[e+1]), (0, 255, 255), 7)

    return img
    
# fahrzeug bewegen
def move_vehicle(lateral_error, vel):
    # steuerung des fahrzeugs anhand der abweichung zur fahrbahnmitte
    steering_angle = -numpy.arctan(lateral_error / 5)
    # konstante geschwindigkeit
    move_cmd.linear.x = vel
    move_cmd.angular.z = steering_angle
    
    cmd_vel.publish(move_cmd)

# haupteil des programms, wird periodisch aufgerufen
def camera_callback(image_data):  
    # Eingangsbild  
    input_image = numpy.frombuffer(image_data.data, dtype=numpy.uint8).reshape(image_data.height, image_data.width, -1)

    # -> schwartz weiss bild
    image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # ROI
    image[0:-1][555:800] = 0
    image[0:-1][0:400] = 0

    # thresholding
    ret, threshold_image = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY)

    # filter fuer edge detection
    edge_image = cv2.GaussianBlur(image, (5, 5), 0)
    # canny edge detection
    edge_image = cv2.Canny(edge_image, 200, 250)

    # kombiniertes Bild
    combined_image = edge_image + threshold_image

    # entfernen der kanten die durch ROI entstanden sind
    combined_image[0:-1][390:412] = 0
    combined_image[0:-1][550:560] = 0

    # keine transformation in vp
    combined_image_bv = combined_image

    # find lane starting points
    lane_opposing_start, lane_left_start, lane_right_start, start_y = find_lane_start(combined_image_bv)

    # sliding window algorithm fuer linke fahrspur
    lane_points_left_x_px, lane_points_left_y_px = sliding_windows(combined_image_bv, lane_left_start,
                                                                               start_y, 4, 10)
    # sliding window algorithm fuer rechte fahrspur
    lane_points_right_x_px, lane_points_right_y_px = sliding_windows(combined_image_bv, lane_right_start,
                                                                                  start_y, 4, 10)
    # sliding window algorithm fuer gegen fahrspur
    lane_points_opposing_x_px, lane_points_opposing_y_px = sliding_windows(combined_image_bv,
                                                                                           lane_opposing_start,
                                                                                           start_y, 4, 10)

    # lane points pixel in meter in vogelperspektive umwandeln
    lane_points_left_x_m, lane_points_left_y_m = Cam.cam_coordinates_to_plane_m(lane_points_left_x_px,
                                                                            lane_points_left_y_px)
    lane_points_right_x_m, lane_points_right_y_m = Cam.cam_coordinates_to_plane_m(lane_points_right_x_px,
                                                                              lane_points_right_y_px)
    lane_points_opposing_x_m, lane_points_opposing_y_m = Cam.cam_coordinates_to_plane_m(lane_points_opposing_x_px,
                                                                                    lane_points_opposing_y_px)

    # Mittellinie interpolieren
    lane_points_left_x_m, lane_points_left_y_m, lane_points_right_x_m, lane_points_right_y_m, lane_points_opposing_x_m, \
            lane_points_opposing_y_m = interpolate_mid_lane(lane_points_left_x_m, lane_points_left_y_m, lane_points_right_x_m,
                                                lane_points_right_y_m, lane_points_opposing_x_m,
                                                lane_points_opposing_y_m)
                                  
    if lane_points_opposing_y_m.size > 20 and lane_points_left_y_m.size > 20 and lane_points_right_y_m.size > 20:
        # klothoide an daten anpassen
        curvature_left, clothoid_x_left, clothoid_y_left = fit_clothoid(lane_points_left_x_m[10:], lane_points_left_y_m[10:])
        curvature_right, clothoid_x_right, clothoid_y_right = fit_clothoid(lane_points_right_x_m[10:], lane_points_right_y_m[10:])
        curvature_opposing, clothoid_x_opposing, clothoid_y_opposing = fit_clothoid(lane_points_opposing_x_m[10:],
                                                                           lane_points_opposing_y_m[10:])
                                                                           
        # Erkannte Linien auf Eingangsbild zeichnen
        cl_x_left_kp, cl_y_left_kp = Cam.plane_m_to_cam_px(clothoid_x_left, clothoid_y_left)
        cl_x_right_kp, cl_y_right_kp = Cam.plane_m_to_cam_px(clothoid_x_right, clothoid_y_right)
        cl_x_opposing_kp, cl_y_opposing_kp = Cam.plane_m_to_cam_px(clothoid_x_opposing, clothoid_y_opposing)
    
        cl_kp_img = numpy.zeros((800, 800, 3), numpy.uint8)
    
        cl_kp_img = draw_lines_on_input(cl_kp_img, cl_x_left_kp, cl_y_left_kp)
        cl_kp_img = draw_lines_on_input(cl_kp_img, cl_x_right_kp, cl_y_right_kp)
        cl_kp_img = draw_lines_on_input(cl_kp_img, cl_x_opposing_kp, cl_y_opposing_kp)
                                                                           
        cl_kp_img = cv2.addWeighted(input_image, 1, cl_kp_img, 0.9, 0)
    
        display_image(cl_kp_img)
                                                                       
                                                                       
        # als array zum publishen fuer rqt plot
        pub_co.publish(curvature_opposing[0])
        pub_cl.publish(curvature_left[0])
        pub_cr.publish(curvature_right[0])
    
        print(curvature_right[0], " ", rospy.get_time())      
        
        # Mittelpunkt der Fahrbahn, relativ zum Fahrzeug
        lateral_error = (lane_points_left_x_m[0] + lane_points_right_x_m[0]) / 2
        move_vehicle(lateral_error, 10)        
    elif lane_points_left_y_m.size > 0 and lane_points_right_y_m.size > 0:
        display_image(input_image)
        lateral_error = (lane_points_left_x_m[0] + lane_points_right_x_m[0]) / 2
        move_vehicle(lateral_error, 10)
    else:
        display_image(input_image)  

# Subscribter zur Kanera in der Gazebo Simulation
rospy.Subscriber("/catvehicle/camera_left/image_raw_left", Image, camera_callback)

if __name__ == '__main__':        
    while not rospy.is_shutdown():            
        rate.sleep() 

