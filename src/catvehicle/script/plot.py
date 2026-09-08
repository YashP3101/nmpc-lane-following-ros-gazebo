#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
csv_file_path = 'lane_detection_data_mpc.csv'
data = pd.read_csv(csv_file_path)

# Print the column names to verify
print(data.columns)

# Ensure the data is clean
data = data.dropna()

# Generate an iteration index
iterations = range(1, len(data) + 1)

# Convert Series to numpy arrays
processing_time = data['Processing_Time'].values
navigation_precision = data['Navigation_Precision'].values
lane_detection_accuracy = data['Lane_Detection_Accuracy'].values

# Plot Iteration vs Processing Time
plt.figure(figsize=(10, 6))
plt.plot(iterations, processing_time, label='Processing Time')
plt.title('Iteration vs Processing Time')
plt.xlabel('Iteration')
plt.ylabel('Processing Time')
plt.legend()
plt.show()

# Plot Iteration vs Navigation Precision
plt.figure(figsize=(10, 6))
plt.plot(iterations, navigation_precision, label='Navigation Precision')
plt.title('Iteration vs Navigation Precision')
plt.xlabel('Iteration')
plt.ylabel('Navigation Precision')
plt.legend()
plt.show()

# Plot Iteration vs Lane Detection Accuracy
plt.figure(figsize=(10, 6))
plt.plot(iterations, lane_detection_accuracy, label='Lane Detection Accuracy')
plt.title('Iteration vs Lane Detection Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Lane Detection Accuracy')
plt.legend()
plt.show()

# Plot Steering Angle vs Time (USING ROS Time)
ros_time = data['Timestamps(ros_time)'].values
ros_time = ros_time - ros_time[0]  
steering_angle = data['steering_angle'].values

fig_steer = plt.figure(figsize=(10, 4))
plt.plot(ros_time, steering_angle, linewidth=1.2)
plt.axhline(0.0, color='k', linewidth=0.8)
plt.xlabel('Time [s]')
plt.ylabel('Steering Angle [rad]')
plt.title('Steering Angle (PPC) vs Time')
plt.tight_layout()
plt.savefig('steering_vs_time_ppcnewfinalruuunnnnnnn5.png', dpi=300)
plt.show()
