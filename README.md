# Autonomous Lane-Following via NMPC in ROS/Gazebo

A real-time robotics simulation stack implementing Nonlinear Model Predictive Control (NMPC) for an autonomous vehicle performing lane-following in Gazebo.

---

## Architecture & Workspace Modules

The workspace is organized into modular ROS packages handling dynamics, sensor simulation, control, and state management:

* **`carbot` / `catvehicle`**: Kinematic and dynamic 4-wheel vehicle simulation models (URDF/Xacro), Ackerman steering kinematics, and Gazebo world environments.
* **`cmdvel2gazebo` & `stepvel`**: Teleoperation and velocity command bridges translating `geometry_msgs/Twist` into direct Gazebo joint-effort/wheel controllers.
* **`control_toolbox`**: Low-level PID controllers and actuation limiters.
* **`obstaclestopper`**: Safety supervision module performing forward distance checks and emergency brake triggering.
* **`sicktoolbox` & `sicktoolbox_wrapper`**: Driver wrapper and scan filtering for SICK 2D laser rangefinders.
* **`velodyne`**: 3D LiDAR point cloud driver and filtering stack (`velodyne_pointcloud`, `velodyne_description`).
* **`log`**: Diagnostics, state-logging, and trajectory history tracking.

---

## Technical Highlights

* **Formulation**: CasADi-based constrained nonlinear optimal control formulation with the IPOPT interior-point solver, solved at 10 Hz over a 20-step prediction horizon and 15-step control horizon (sampling time $T_s = 0.1$ s).
* **Numerical Integration**: Fourth-order Runge-Kutta (RK4) discretization for high-fidelity multi-step state prediction, implemented as an equality constraint within the CasADi optimization.
* **Control Budget**: NMPC solver averaged ~47 ms per control cycle, comfortably within the 100 ms real-time budget imposed by the 10 Hz control loop.
* **Kinematics**: Kinematic bicycle model with non-holonomic constraints and state boundary conditions.
  * **Velocity & Limits**: Constant longitudinal velocity ($v = 10\text{ m/s}$) with front-wheel steering input saturation ($\delta \in [-0.6, 0.6]\text{ rad}$).
* **Perception Pipeline**: Reused the existing LiDAR-camera sensor fusion pipeline (white-point filtering, DBSCAN clustering, RANSAC line fitting) from the prior baseline framework. Since the raw slope/intercept outputs were noisy frame-to-frame, an Exponential Moving Average (EMA) filter was added and fine-tuned on top of the existing pipeline to stabilize the lane-boundary estimates before feeding them to the controller.

---

## Prerequisites & Installation

### Environment
* **OS**: Ubuntu 20.04 LTS
* **ROS**: Noetic Ninjemys
* **Physics Engine**: Gazebo 11

### Build Instructions
```bash
# Clone the repository
mkdir -p ~/nmpc_ws/src
cd ~/nmpc_ws/src
git clone https://github.com/YashP3101/nmpc-lane-following-ros-gazebo.git .

# Install Python and ROS dependencies
cd ~/nmpc_ws
pip3 install -r src/requirements.txt
# or use: pip3 install -r src/noetic_requirements.txt
bash src/requirements.sh

# Resolve system dependencies and compile
rosdep install --from-paths src --ignore-src -r -y
catkin_make
source devel/setup.bash
```

---

## Asset Configuration (Mesh Files)

To maintain an efficient repository size, large CAD/simulation visual meshes (`rav4.dae`) are excluded:
* Place `rav4.dae` into `src/catvehicle/urdf/` prior to high-fidelity visual simulation runs.
* Collision geometry and physics are fully retained via native geometric primitives.

---

## Running the Simulation

Follow these steps in separate terminals (ensure your workspace is sourced via `source devel/setup.bash` in each terminal):

---

### 1. Controller Selection

Open `src/catvehicle/scripts/right_lane_follower.py` and set the controller mode:

* **NMPC Execution:** Set `use_mpc = True` (ensure `/odom` subscriber is active)
* **Pure Pursuit Execution:** Set `use_mpc = False` (keep `/odom` subscriber commented out)

---

### 2. Execution Steps (Step-by-Step)

### Step 1: Launch the Gazebo Simulation World

```bash
roslaunch carbot_gazebo carbot_world.launch
```

### Step 2: Spawn the CATVehicle Model

```bash
roslaunch catvehicle catvehicle_spawn.launch robot:=catvehicle X:=2.0 Y:=1.5 yaw:=0
```

### Step 3: Launch Sensor Fusion (LiDAR & Camera)

```bash
rosrun catvehicle fusion2pc12.py
```

### Step 4: Launch Test Multiplexer

```bash
rosrun catvehicle test_multiplexer_v_2.py
```

### Step 5: Launch Test Injector

```bash
rosrun catvehicle test_injector_v_2.py
```

### Step 6: Run Lane Follower & Controller Node

```bash
rosrun catvehicle right_lane_follower.py
```

### Step 7: Toggle Sensor Data Stream

```bash
rosservice call /toggle_test_data "data: false"
```

> **Note:** To test synthetic point clouds or sensor interruption, pass `"data: true"`.

---

### 3. Alternative Spawn Scenarios (Track Testing)

To test specific track segments directly:

* **Before Left Turn:**
  ```bash
  roslaunch catvehicle catvehicle_spawn.launch robot:=catvehicle X:=305.0 Y:=1.5 yaw:=0
  ```

* **Before Right Turn:**
  ```bash
  roslaunch catvehicle catvehicle_spawn.launch robot:=catvehicle X:=333.725673 Y:=228.083521 yaw:=2.699196
  ```

* **Before Track End Point:**
  ```bash
  roslaunch catvehicle catvehicle_spawn.launch robot:=catvehicle X:=27.5 Y:=680.0 yaw:=0
  ```

---

### 4. Post-Simulation Visualization & Plots

Generate steering smoothness, steering angle over time, and actual vehicle path vs. reference trajectory plots:

```bash
rosrun catvehicle open_plot.py
```

---

### Author

* **Degree:** M.Sc. Mechatronics, University of Siegen
* **Focus:** Advanced Control Systems, Model Predictive Control (MPC/NMPC), ROS/Gazebo Simulation

### Acknowledgments

* **Simulation Framework & Dynamics:** Vehicle simulation models and Gazebo dynamics adapt components from the open-source CATVehicle testbed (Rahul Bhadani et al., University of Arizona).
* **Baseline Framework:** Built upon the perception pipeline and baseline Pure Pursuit Controller stack developed by Krishna Gopal Kundu (*Robust Lane Detection and Navigation for Autonomous Vehicles with Sensor Fusion*, 2024). This thesis retains that perception pipeline unchanged in its core algorithms, replaces the Pure Pursuit Controller with the NMPC described above, and adds an EMA filtering stage to stabilize the perception output.
