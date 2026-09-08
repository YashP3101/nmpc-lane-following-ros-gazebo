#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Twist
import sensor_msgs.point_cloud2 as pc2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
from collections import deque
from nav_msgs.msg import Odometry
from kinematic_nmpc import CATvehicleNMPC
import time


class LaneFollower:
    def __init__(self, init_node=True):
        if init_node:
            rospy.init_node('lane_detection', anonymous=True)
        self.point_cloud_sub = rospy.Subscriber('/output_point_cloud2_new', PointCloud2, self.point_cloud_callback)
        self.cmd_vel_pub = rospy.Publisher('/catvehicle/cmd_vel', Twist, queue_size=100)
        self.rate = rospy.Rate(10)  # 10 Hz


        # "============================================"
        # Model Predictive Controller
        self.mpc_controller = CATvehicleNMPC()

        # Kinematic NMPC states:[x, y, psi]
        self.current_state = np.zeros(3)

        # Constant speed
        self.constant_speed = 10.0

        # Quick steering sign switch (in case Gazebo sign is opposite)
        # +1 means "use MPC delta as-is"
        # -1 means "flip sign"
        self.steer_sign = +1.0

        # Controller selection mpc or existing ppc
        self.use_mpc = True  # Set False for Pure Pursuit Controller

        self.boundary_lines = None

        self.vehicle_path = [] 
        self.last_ref_traj = None

        self.boundary_history = []
        self.ref_traj_history = []

        # Odom subscriber
        self.odom_sub = rospy.Subscriber('/catvehicle/odom', Odometry, self.odom_callback)
        # "============================================"
        
        # for EMA
        self.smoothed_boundary_lines = {}  # key: lane index, value: (m, b)

        self.boundary_ema_alpha_mpc = 0.40  # tune: 0.1=smooth, 0.4=responsive
        self.boundary_ema_alpha_ppc = 0.40   # tune: 0.1=smooth, 0.4=responsive

        self.latest_points = None
        self.target_path = None
        self.base_look_ahead_distance = 5.5  # Base look-ahead distance in meters
        self.look_ahead_distance = self.base_look_ahead_distance
        self.lane_detected = False
        self.lane_markings_visible = True  # Flag to indicate if lane markings are visible

        # Setup Matplotlib for interactive mode
        plt.ion()
        self.fig, self.ax = plt.subplots()

        # History buffer for lane clusters
        self.lane_history = deque(maxlen=10)

        self.target_path_buffer = deque(maxlen=2)

    def point_cloud_callback(self, msg):
        point_list = []
        for point in pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "rgba")):
            x, y, z, rgba = point
            r = int((rgba >> 16) & 0xFF)
            g = int((rgba >> 8) & 0xFF)
            b = int(rgba & 0xFF)
            # Threshold for white color and range 0 to 10 meters in front, y within -2 to 4
            if r > 200 and g > 200 and b > 200 and 0 <= x <= 10 and -2 <= y <= 4:    
                                point_list.append((x, y, z))
        
        rospy.loginfo(f"Filtered {len(point_list)} white points")
        if len(point_list) == 0:
            self.lane_markings_visible = False
        else:
            self.lane_markings_visible = True
        self.latest_points = point_list if len(point_list) > 0 else self.latest_points

    def process_point_list(self):

        if self.latest_points is None or not self.lane_markings_visible:
            self.lane_detected = False
            return
        
        points = np.array(self.latest_points)
        if points.size == 0:
            rospy.loginfo("No white points found")
            self.lane_detected = False
            return
        
        points_2d = points[:, :2]

        # Adjusting delta_interceptSCAN parameters
        eps_value = 2.65 # Adjusted epsilon for more sensitive clustering
        min_samples_value = 4  # Ensure adequate sample size
        clustering = DBSCAN(eps=eps_value, min_samples=min_samples_value).fit(points_2d)
        labels = clustering.labels_
        rospy.loginfo(f"Number of clusters found: {len(set(labels)) - (1 if -1 in labels else 0)}")
        rospy.loginfo(f"delta_interceptSCAN parameters - eps: {eps_value}, min_samples: {min_samples_value}")

        lanes = self.extract_lanes(points_2d, labels)
        
        if len(lanes) >= 1:
            lanes = sorted(lanes, key=len, reverse=True)[:2]  # Take the largest clusters
            self.lane_history.append(lanes)
            boundary_lines = self.fit_boundary_lines(lanes)
            self.boundary_lines = boundary_lines
            self.target_path = self.calculate_target_path(boundary_lines)
            self.lane_detected = True
            # self.update_visualization(lanes, boundary_lines, self.target_path)
            if not self.use_mpc:
                self.update_visualization(lanes, boundary_lines, self.target_path)
        else:
            rospy.loginfo("Expected at least 1 lane, but found none.")
            self.lane_detected = False

            # If fewer than 1 lane is detected, use the most recent valid lanes from history
            if len(self.lane_history) > 0:
                lanes = self.lane_history[-1]
                boundary_lines = self.fit_boundary_lines(lanes)
                self.boundary_lines = boundary_lines
                self.target_path = self.calculate_target_path(boundary_lines)
                # self.update_visualization(lanes, boundary_lines, self.target_path)
                if not self.use_mpc:
                    self.update_visualization(lanes, boundary_lines, self.target_path)


    def extract_lanes(self, points_2d, labels):
        lanes = []
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise

            class_member_mask = (labels == label)
            xy = points_2d[class_member_mask]

            # Ensure a reasonable number of points for a lane and validate lane
            if len(xy) > 6 and self.is_valid_lane(xy):
                lanes.append(xy)

        return lanes

    def is_valid_lane(self, xy):
        # Define criteria for a valid lane
        if len(xy) < 6:
            return False

        # Check the spread along the x-axis to ensure it has a reasonable length
        x_min, x_max = np.min(xy[:, 0]), np.max(xy[:, 0])
        if (x_max - x_min) < 2.0:  # Lane length should be at least 2 meters
            return False

        # Fit a line and check the deviation of points from the line
        line_model = RANSACRegressor()
        X = xy[:, 0].reshape(-1, 1)
        y = xy[:, 1]
        line_model.fit(X, y)
        y_pred = line_model.predict(X)
        residuals = np.abs(y - y_pred)

        if np.mean(residuals) > 0.5:  # Check if average deviation is within 0.5 meters
            return False

        return True

    # def fit_boundary_lines(self, lanes):
    #     boundary_lines = []
    #     for lane in lanes:
    #         # Fit a RANSAC model with linear regression
    #         X = lane[:, 0].reshape(-1, 1)
    #         y = lane[:, 1]
    #         ransac = RANSACRegressor()
    #         ransac.fit(X, y)
            
    #         # Extract the linear coefficients
    #         slope = ransac.estimator_.coef_[0]
    #         intercept = ransac.estimator_.intercept_
    #         boundary_lines.append((slope, intercept))

    #     return boundary_lines
    
    def fit_boundary_lines(self, lanes):
        boundary_lines = []
        for i, lane in enumerate(lanes):
            X = lane[:, 0].reshape(-1, 1)
            y = lane[:, 1]
            ransac = RANSACRegressor(min_samples=5, residual_threshold=0.3)
            # ransac = RANSACRegressor()
            ransac.fit(X, y)
            
            slope = ransac.estimator_.coef_[0]
            intercept = ransac.estimator_.intercept_

            # EMA FILTER
            if i in self.smoothed_boundary_lines:
                prev_m, prev_b = self.smoothed_boundary_lines[i]
                alpha = self.boundary_ema_alpha_mpc if self.use_mpc else self.boundary_ema_alpha_ppc
                slope = alpha * slope + (1 - alpha) * prev_m
                intercept = alpha * intercept + (1 - alpha) * prev_b
            
            self.smoothed_boundary_lines[i] = (slope, intercept)
            
            boundary_lines.append((slope, intercept))

        return boundary_lines
        

    def calculate_target_path(self, boundary_lines):
        if len(boundary_lines) == 0:
            return None  # Need at least one boundary line

        else:
            # Determine which boundary line is right and which is left
            if len(boundary_lines) == 2:
                if boundary_lines[0][1] > boundary_lines[1][1]:
                    right_lane = boundary_lines[0]
                    left_lane = boundary_lines[1]
                else:
                    right_lane = boundary_lines[1]
                    left_lane = boundary_lines[0]
            else:
                # Only one lane detected, treat it as the right lane
                right_lane = boundary_lines[0]
                left_lane = None

            # Hard code the target path using the right lane's slope and adjusted intercept
            if right_lane is not None:
                target_slope = right_lane[0]
                target_intercept = right_lane[1] - 1.5 / np.sqrt(1 + target_slope**2)
            else:
                target_slope = left_lane[0]
                target_intercept = left_lane[1] + 1.5 / np.sqrt(1 + target_slope**2)

            # Store the calculated target path in the buffer
            self.target_path_buffer.append((target_slope, target_intercept))

            # Compute the smoothed target path using the moving average
            avg_slope = np.mean([path[0] for path in self.target_path_buffer])
            avg_intercept = np.mean([path[1] for path in self.target_path_buffer])

            return (avg_slope, avg_intercept)

    # def update_visualization(self, lanes, boundary_lines, target_path):
    #     self.ax.clear()

    #     # Plot lanes
    #     for lane in lanes:
    #         self.ax.plot(lane[:, 0], lane[:, 1], 'o', label='Current Lane Points')

    #     # Plot boundary lines
    #     for (slope, intercept), lane in zip(boundary_lines, lanes):
    #         x_vals = np.linspace(0, 10, 100)
    #         y_vals = slope * x_vals + intercept
    #         self.ax.plot(x_vals, y_vals, '--', label='Boundary Line')

    #     # Plot target path
    #     if target_path is not None:
    #         target_slope, target_intercept = target_path
    #         x_vals = np.linspace(0, 10, 100)
    #         y_vals = target_slope * x_vals + target_intercept
    #         self.ax.plot(x_vals, y_vals, 'r', linewidth=2, label='Target Path')

    #     # Plot the label for buffered lane points once
    #     if len(self.lane_history) > 0:
    #         # Plot the label only for the first point set
    #         first_lane_set = self.lane_history[0]
    #         if len(first_lane_set) > 0:
    #             first_lane = first_lane_set[0]
    #             self.ax.plot(first_lane[:, 0], first_lane[:, 1], 'x', alpha=0.3, label='Buffered Lane Points')

    #     # Plot buffered lane points from history without label
    #     for lanes_in_history in self.lane_history:
    #         for lane in lanes_in_history:
    #             self.ax.plot(lane[:, 0], lane[:, 1], 'x', alpha=0.3)

    #     self.ax.set_xlim([0, 10])
    #     self.ax.set_ylim([-5, 5])
    #     self.ax.set_aspect('equal', adjustable='box')
    #     self.ax.set_title("Detected Lanes")
    #     self.ax.set_xlabel("X")
    #     self.ax.set_ylabel("Y")
    #     self.ax.grid(True)
    #     self.ax.legend()

    #     plt.draw()
    #     plt.pause(0.001)


    def calculate_look_ahead_distance(self, speed):
        # Adjust the look-ahead distance based on the current speed of the vehicle
        return max(self.base_look_ahead_distance, speed / 3.6 * 1.5)  # Adjust the factor as needed

    def move_vehicle_ppc(self):     # PURE PURSUIT CONTROL
        if not self.lane_markings_visible:
            # Stop the vehicle if no white lane markings are visible
            twist = Twist()
            twist.linear.x = 0
            twist.angular.z = 0
            self.cmd_vel_pub.publish(twist)
            return

        if not self.lane_detected or self.target_path is None:
            # Use the most recent valid target path if lanes are not detected
            if len(self.lane_history) > 0:
                self.target_path = self.calculate_target_path(self.fit_boundary_lines(self.lane_history[-1]))
            if self.target_path is None:
                # Stop the vehicle if no lanes are detected at all
                twist = Twist()
                twist.linear.x = 0
                twist.angular.z = 0
                self.cmd_vel_pub.publish(twist)
                return

        target_slope, target_intercept = self.target_path

        # Get current speed of the vehicle (assume constant speed of 10 m/s for simplicity)
        current_speed = 10

        # Calculate dynamic look-ahead distance based on speed
        self.look_ahead_distance = self.calculate_look_ahead_distance(current_speed)

        # Get current position of the vehicle (Assuming it is at origin (0,0))
        current_x = 0
        current_y = 0

        # Calculate the look-ahead point on the target path
        look_ahead_x = current_x + self.look_ahead_distance
        look_ahead_y = target_slope * look_ahead_x + target_intercept

        # Validate the look-ahead point calculation
        if look_ahead_y < -5 or look_ahead_y > 5:
            rospy.loginfo("Invalid look-ahead point, stopping vehicle")
            twist = Twist()
            twist.linear.x = 0
            twist.angular.z = 0
            self.cmd_vel_pub.publish(twist)
            return

        # Calculate the steering angle using Pure Pursuit formula
        dx = look_ahead_x - current_x
        dy = look_ahead_y - current_y
        Ld = np.sqrt(dx**2 + dy**2)
        alpha = np.arctan2(dy, dx)
        steering_angle = 2 * dy / (Ld**2)

        # Adjust steering angle for right turns
        if target_slope < 0:  # If target slope is negative, indicating a right turn
            steering_angle = -abs(steering_angle)  # Ensure the steering angle is negative for right turns
        else:
            steering_angle = abs(steering_angle)  # Ensure the steering angle is positive for left turns

        twist = Twist()
        twist.linear.x = current_speed  # Use current speed
        twist.angular.z = steering_angle  # Angular velocity based on Pure Pursuit output
        self.cmd_vel_pub.publish(twist)


    # "====================================================="
    # "-------------MODEL PREDICTIVE CONTROL----------------"
    # "====================================================="

    def quat_to_yaw(self, quat):
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        return np.arctan2(siny_cosp, cosy_cosp)
    
    
    def _wrap_angle(self, a):
        return (a + np.pi) % (2 * np.pi) - np.pi
    
    
    def calculate_centerline(self, boundary_lines, lane_half_width=1.5):
        
        # Returns centerline (m_c, b_c) in SAME frame (base_link) as boundary_lines
        # boundary_lines: [(m1,b1), (m2,b2)] ideally.
        # If only one line: shift it by lane_half_width towards center from right boundary
        
        if len(boundary_lines) == 0:
            return None

        # Decide left/right boundary lane using intercept
        if len(boundary_lines) >= 2:
            (m1, b1), (m2, b2) = boundary_lines[0], boundary_lines[1]

            # bigger intercept ==> right lane (m:slope, b:intercept)
            if b1 > b2:
                right_lane = (m1, b1)
                left_lane  = (m2, b2)
            else:
                right_lane = (m2, b2)
                left_lane  = (m1, b1)

            mR, bR = right_lane
            mL, bL = left_lane

            # Centerline as average
            mC = 0.5 * (mL + mR)
            bC = 0.5 * (bL + bR)

            return (mC, bC)

        # Only one lane detected: shift it by lane_half_width to approximate center
        m, b = boundary_lines[0]
        mC = m

        # perpendicular distance offset for line y = m x + b:
        # shift b by +/- lane_half_width * sqrt(1+m^2)
        # We assume this single detected line is right lane
        bC = b - lane_half_width * np.sqrt(1.0 + m*m)

        return (mC, bC)
    
    
    def update_mpc_visualization(self):
        # MPC visualization during run, showing:
        # - Boundary lines (left & right lane markings) 
        # - MPC reference trajectory (planned path)
        # - Vehicle current position with heading
        # - Vehicle's actual traveled path
        # All in odom frame
        
        if not self.use_mpc:
            return

        self.ax.clear()

        # Current vehicle position in odom
        x_vehicle, y_vehicle, psi_vehicle = self.current_state

        # 1.Plot boundary lines (left & right lane markings)
        if hasattr(self, 'bl_lines_odom') and len(self.bl_lines_odom) > 0:
            colors = ['grey', 'grey']
            labels = ['Left Boundary', 'Right Boundary']
            
            for idx, line_points in enumerate(self.bl_lines_odom):
                color = colors[idx] if idx < len(colors) else 'gray'
                label = labels[idx] if idx < len(labels) else f'Boundary {idx+1}'
                self.ax.plot(line_points[:, 0], line_points[:, 1], 
                            color=color, linestyle='-', linewidth=1.0, label=label)

        # 2.Plot MPC reference trajectory
        if self.last_ref_traj is not None:
            self.ax.plot(self.last_ref_traj[:, 0], self.last_ref_traj[:, 1], 
                        'r-', linewidth=1.0, marker=' ', markersize=2, 
                        markevery=2, label='MPC Reference Path')

        # 3.Plot vehicle's actual traveled path
        if len(self.vehicle_path) > 2:
            path_x, path_y = zip(*self.vehicle_path)
            self.ax.plot(path_x, path_y, 'b-', linewidth=1.2, 
                        alpha=0.7, label='Actual Vehicle Path')

        # 4.Plot current vehicle position
        self.ax.plot(x_vehicle, y_vehicle, 'ro', markersize=5, 
                    label='Vehicle Position', zorder=5)
        
        # Vehicle heading arrow
        arrow_length = 0.5
        dx_arrow = arrow_length * np.cos(psi_vehicle)
        dy_arrow = arrow_length * np.sin(psi_vehicle)
        self.ax.arrow(x_vehicle, y_vehicle, dx_arrow, dy_arrow,
                    head_width=0.5, head_length=0.3, fc='red', ec='red', 
                    linewidth=2, zorder=5)

        # Plot settings (dynamic view following vehicle)
        self.ax.set_aspect('equal', adjustable='box')
        plot_size = 30  # window size
        self.ax.set_xlim(x_vehicle - plot_size/2, x_vehicle + plot_size/2)
        self.ax.set_ylim(y_vehicle - plot_size/2, y_vehicle + plot_size/2)
        self.ax.grid(True, alpha=0.1)
        self.ax.set_xlabel('X (odom) [m]')
        self.ax.set_ylabel('Y (odom) [m]')
        self.ax.set_title(f'NMPC Lane Following')
        self.ax.legend(loc='upper left', fontsize=8)

        plt.draw()
        plt.pause(0.001)



    def odom_callback(self, msg):
        # speed from odom
        v = msg.twist.twist.linear.x
        self.current_speed = v if v > 0.1 else self.constant_speed

        # pose in odom
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = self.quat_to_yaw(msg.pose.pose.orientation)

        self.current_state[0] = x
        self.current_state[1] = y
        self.current_state[2] = yaw

        self.odom_initialized = True

        rospy.loginfo_throttle(
            1.0,
            f"[ODOM_STATE] x={x:+.2f} | y={y:+.2f} | yaw={yaw:+.2f} | v={self.current_speed:+.2f}"
        )



    def move_vehicle_mpc(self):
        # Safety: lane visibility
        if not self.lane_markings_visible:
            rospy.logwarn("MPC: no lane markings -> STOP")
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)
            return

        if not self.lane_detected or self.target_path is None:
            if len(self.lane_history) > 0:
                self.target_path = self.calculate_target_path(self.fit_boundary_lines(self.lane_history[-1]))
                rospy.logwarn("MPC: using lane history fallback")
            if self.target_path is None:
                rospy.logwarn("MPC: no target path -> STOP")
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_vel_pub.publish(twist)
                return

        # Safety: odom must be ready
        if not getattr(self, "odom_initialized", False):
            rospy.logwarn_throttle(1.0, "MPC: waiting for /catvehicle/odom: STOP")
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)
            return

        #  Current odom pose
        x0 = float(self.current_state[0])
        y0 = float(self.current_state[1])
        psi0 = float(self.current_state[2])

        # Build reference trajectory from CENTERLINE in base_link frame, then tf to odom
        boundary_lines = self.boundary_lines
        if boundary_lines is None:
            return
        
        x_offset = 2.466
        bl_lines = [(m, b - m*x_offset) for (m, b) in boundary_lines]
        center = self.calculate_centerline(bl_lines, lane_half_width=1.5)

        if center is None:
            rospy.logwarn_throttle(1.0, "MPC: centerline not available: STOP")
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)
            return  

        mC, bC = center

        # Speed adaptation based on turn (to improve turns)
        turn_curvature = abs(mC)          # turn strength, larger slope = sharper turn
        v_max = float(self.constant_speed)  # 10 m/s
        v_min = 7.5                 
        vref = v_max / (1.0 + 3.0 * turn_curvature)
        vref = float(np.clip(vref, v_min, v_max))
        self.last_vref = vref

        N  = self.mpc_controller.pred_horizon
        dt = self.mpc_controller.time_step
    
        cy = np.cos(psi0)
        sy = np.sin(psi0)

        ref_traj = np.zeros((N, 3), dtype=float)

        ref_traj[0,0] = x0
        ref_traj[0,1] = y0
        ref_traj[0,2] = psi0

        for k in range(N):
            s = vref* k * dt

            # centerline in base_link
            x_bl = s
            y_bl = mC * x_bl + bC
            psi_bl = np.arctan(mC)  

            # base_link to odom
            x_o = x0 + cy * x_bl - sy * y_bl
            y_o = y0 + sy * x_bl + cy * y_bl
            psi_o = self._wrap_angle(psi0 + psi_bl)

            ref_traj[k] = [x_o, y_o, psi_o]

        
        self.last_ref_traj = ref_traj.copy()
        self.vehicle_path.append((float(x0), float(y0)))

        # Transform boundary lines to odom (purpose:for visualization of plot/graph)
        cos_psi = np.cos(psi0)
        sin_psi = np.sin(psi0)

        self.bl_lines_odom = []
        for (m_bl, b_bl) in bl_lines:
            # Create line points in base_link
            x_bl_vals = np.linspace(0, 20, 30)   
            y_bl_vals = m_bl * x_bl_vals + b_bl
            
            # Transform to odom frame
            x_odom = x0 + cos_psi * x_bl_vals - sin_psi * y_bl_vals
            y_odom = y0 + sin_psi * x_bl_vals + cos_psi * y_bl_vals
            
            self.bl_lines_odom.append(np.column_stack([x_odom, y_odom]))

        # Also store centerline in odom
        x_center_bl = np.linspace(0, 15, 30)
        y_center_bl = mC * x_center_bl + bC
        x_center_odom = x0 + cos_psi * x_center_bl - sin_psi * y_center_bl
        y_center_odom = y0 + sin_psi * x_center_bl + cos_psi * y_center_bl
        self.centerline_odom = np.column_stack([x_center_odom, y_center_odom])

        self.boundary_history.append([line.copy() for line in self.bl_lines_odom])
        self.ref_traj_history.append(self.last_ref_traj.copy())

        t_start = rospy.Time.now().to_sec()
        # TOTAL MPC PIPELINE START (detection → solve → publish)
        _t_pipeline_start = time.perf_counter()
        t0 = time.perf_counter();  # MPC solve

        # Solve NMPC delta only
        try:
            steering_rad = self.mpc_controller.compute_controls(
                current_state=self.current_state,
                reference_traj=ref_traj,
                v_ref=vref
            )
        except Exception as e:
            rospy.logerr(f"MPC solve failed: {str(e)}")
            return

        t_end = rospy.Time.now().to_sec()
        rospy.loginfo(f"MPC_TIME_MS={(time.perf_counter()-t0)*1000:.1f}")

        # Apply sign
        steering_rad = self.steer_sign * steering_rad
        steering_gazebo = np.clip(steering_rad, -0.55, 0.55) 
        
        # Publish
        twist = Twist()
        twist.linear.x = vref
        twist.angular.z = steering_gazebo
        self.cmd_vel_pub.publish(twist)

        # TOTAL PIPELINE TIME
        _t_pipeline_ms = (time.perf_counter() - _t_pipeline_start) * 1000
        rospy.loginfo(f"Total MPC Pipeline: {_t_pipeline_ms:.2f} ms")

        # Visualization
        if self.use_mpc:
            self.update_mpc_visualization()

        len(self.vehicle_path) > 15
        #     self.vehicle_path.pop(0)

        rospy.loginfo(
            f"[MPC_ODOM] mC={mC:+.3f} bC={bC:+.3f} | "
            f"psi0={psi0:+.3f} psi_ref={ref_traj[0,2]:+.3f} | steer={steering_rad:+.3f}"
        )

        rospy.loginfo(f"MPC_TIME_MS={(t_end - t_start)*1000:.2f}")


    def run(self):
        
        self.mpc_controller.reset_warmstart() 
        while not rospy.is_shutdown():
            self.process_point_list()
            if self.use_mpc:
                self.move_vehicle_mpc()
            else:
                self.move_vehicle_ppc()
            self.rate.sleep()
        

if __name__ == '__main__':
    lane_follower = LaneFollower()
    lane_follower.run()
