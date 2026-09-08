#! /bin/bash
gnome-terminal -- bash -c "killall gzserver" &
sleep 1 &&
echo ' Gazeobo starts running ... '

gnome-terminal -- bash -c "roslaunch carbot_gazebo carbot_world.launch" &
sleep 25 &&
echo ' Cat vehicle model is loading ... '

gnome-terminal -- bash -c "roslaunch catvehicle catvehicle_spawn.launch robot:=catvehicle X:=2 Y:=1.5 \yaw:=0" &
sleep 10 &&

gnome-terminal -- bash -c "./lane_detection_kp.py" &
sleep 5 &&
echo ' Fahrbahnmarkierungserkennung in Vogelperspektive wird ausgeführt ... '

echo ' '
echo ' finish!'
echo ' '
