#!/bin/bash

# Wait for Gazebo to launch
sleep 10

# Spawn the vehicle
gnome-terminal -- bash -c "roslaunch catvehicle catvehicle_spawn.launch robot:=catvehicle X:=2.0 Y:=1.5 \yaw:=0; exec bash" &

# Wait before running fusion.py
sleep 10
gnome-terminal -- bash -c "rosrun catvehicle fusion2pcl2.py; exec bash" &

# Wait before running multiplexer.py
sleep 10
gnome-terminal -- bash -c "rosrun catvehicle test_multiplexer_v_2.py; exec bash" &

# Wait before running navigation.py
sleep 10
gnome-terminal -- bash -c "rosrun catvehicle right_lane_follower.py; exec bash" &

# Wait before running rosservice call
# sleep 10
# gnome-terminal -- bash -c "rosservice call /toggle_test_data 'data: false'; exec bash"
