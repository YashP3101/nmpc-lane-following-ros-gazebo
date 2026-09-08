#!/usr/bin/env python3
#---------------------------------------------------------
# Kinematic NMPC Controller for CATvehicle Lane Following

# Model: Kinematic bicycle
# RK4 descritization
# States:   [x, y, psi]
# Control:  [delta]  (steering)
# Speed:    constant v_ref (10 m/s)
# ----------------------------------------------------------

import numpy as np
import casadi as ca
import traceback
import time

class CATvehicleNMPC:

    def __init__(self):
        
        # Vehicle geometry
        self.front_axle_dist = 1.2   # m (CG -> front axle)
        self.rear_axle_dist  = 1.6   # m (CG -> rear axle)
        self.wheelbase = self.front_axle_dist + self.rear_axle_dist  # L
        
        # Hard constraints
        self.max_steering = 0.6      # 0.6 rad (~34°)
        
        # MPC parameters
        self.pred_horizon = 20       # Prediction steps (N)
        self.ctrl_horizon = 15       # Control steps (M) (M<=N)
        self.time_step = 0.1         # sec (dt)

        # Cache for warm start
        self._last_states = None
        self._last_controls = None


        # Weights
        # States: [x, y, psi]
        # Control: [delta]
        # Plus smoothness term on delta change
        self.Q  = ca.DM(np.diag([5.0, 27.0, 15.0]))   # track x,y,psi
        self.R  = ca.DM(np.diag([500.0]))             # steering effort   
        self.Rd = ca.DM(np.diag([1100.0]))            # steering smoothness 

        # Build optimizer
        self.optimizer = self._build_optimizer()


    def _build_optimizer(self):
        # Create CasADi optimization problem for kinematic NMPC
        opti = ca.Opti()

        # Decision variables
        self.states = opti.variable(3, self.pred_horizon + 1)
        # Controls: [delta]
        self.controls = opti.variable(1, self.ctrl_horizon)

        # Initial guesses
        opti.set_initial(self.states, 0)
        opti.set_initial(self.controls, 0)

        # Parameters:
        # x0 (3) + reference trajectory (3*N) + v_ref (1)
        self.params = opti.parameter(3 + 3 * self.pred_horizon + 1)

        x0 = self.params[:3]
        ref_flat = self.params[3:3 + 3 * self.pred_horizon]
        v_ref = self.params[-1]

        ref_trajectory = ca.reshape(ref_flat, 3, self.pred_horizon)   # shape (3, N)


        # Cost function 
        cost = 0
        opti.subject_to(self.states[:, 0] == x0)

        for k in range(self.pred_horizon):
            xk = self.states[:, k]

            # Hold last control after control horizon
            if k < self.ctrl_horizon:
                uk = self.controls[:, k]
            else:
                uk = self.controls[:, -1]

            # Tracking error
            e = xk - ref_trajectory[:, k]
            cost += e.T @ self.Q @ e

            # Steering effort
            cost += uk.T @ self.R @ uk

            # Smoothness (delta rate)
            if k > 0:
                if k < self.ctrl_horizon:
                    uk_prev = self.controls[:, k - 1]
                else:
                    uk_prev = self.controls[:, -1]
                duk = uk - uk_prev
                cost += duk.T @ self.Rd @ duk

        # Terminal cost
        eN = self.states[:, self.pred_horizon] - ref_trajectory[:, self.pred_horizon-1]
        cost += 3.0 * eN.T @ self.Q @ eN

        opti.minimize(cost)


        # Dynamics (RK4 integration)
        for k in range(self.pred_horizon):
            xk = self.states[:, k]

            if k < self.ctrl_horizon:
                uk = self.controls[:, k]
            else:
                uk = self.controls[:, -1]

            k1 = self._vehicle_dynamics(xk, uk, v_ref)
            k2 = self._vehicle_dynamics(xk + 0.5 * self.time_step * k1, uk, v_ref)
            k3 = self._vehicle_dynamics(xk + 0.5 * self.time_step * k2, uk, v_ref)
            k4 = self._vehicle_dynamics(xk + self.time_step * k3, uk, v_ref)

            x_next = xk + (self.time_step / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            opti.subject_to(self.states[:, k+1] == x_next)

        # Constraints
        opti.subject_to(opti.bounded(-self.max_steering, self.controls, self.max_steering))

        # IPOPT Solver
        solver_options = {
            "ipopt.max_iter": 200,
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.acceptable_tol": 1e-3,
            "ipopt.acceptable_obj_change_tol": 1e-3,
            "ipopt.warm_start_init_point": "yes",
        }
        opti.solver("ipopt", solver_options)

        return opti
    

    # Reset warm start to neutral position
    def reset_warmstart(self):
        self._last_states = None
        self._last_controls = None
        print("[KIN_MPC] Warm start reset ✓")
    


    # Kinematic bicycle dynamics
    def _vehicle_dynamics(self, state: ca.SX, control: ca.SX, v_ref: ca.SX) -> ca.SX:
    
        # NOTE: about steering sign:
        # - Standard model uses +delta ==> left turn.
        # - If Gazebo expects opposite sign, flip sign here (commented line below)
    
        x = state[0]
        y = state[1]
        psi = state[2]

        delta = control[0]

        x_dot = v_ref * ca.cos(psi)
        y_dot = v_ref * ca.sin(psi)

        # Standard:
        psi_dot = (v_ref / self.wheelbase) * ca.tan(delta)

        # If later steering sign is reversed in Gazebo, use this instead:
        # psi_dot = -(v_ref / self.wheelbase) * ca.tan(delta)

        return ca.vertcat(x_dot, y_dot, psi_dot)


    def compute_controls(self, current_state: np.ndarray, reference_traj: np.ndarray, v_ref: float = 10.0) -> float:
        
        # Solve kinematic NMPC.

        # Args:
        #     current_state: [x, y, psi] (shape: (3,))
        #     reference_traj: reference trajectory over horizon (shape: (N, 3))
        #                    each row: [x_ref, y_ref, psi_ref]
        #     v_ref: constant speed (10 m/s)

        # Returns:
        #     steering delta (rad)
        
        N = self.pred_horizon
        assert current_state.shape == (3,), f"Expected current_state (3,), got {current_state.shape}"
        assert reference_traj.shape == (N, 3), f"Expected reference_traj ({N},3), got {reference_traj.shape}"

        # Safety checks
        if np.any(np.isnan(current_state)) or np.any(np.isinf(current_state)):
            print("[KIN_MPC_CRASH] NaN/Inf in current_state!")
            return 0.0
        if np.any(np.isnan(reference_traj)) or np.any(np.isinf(reference_traj)):
            print("[KIN_MPC_CRASH] NaN/Inf in reference_traj!")
            return 0.0

        # Warm start
        if self._last_states is not None:
            self.optimizer.set_initial(self.states, self._last_states)
        if self._last_controls is not None:
            self.optimizer.set_initial(self.controls, self._last_controls)

        # Pack parameters
        ref_colmajor = reference_traj.T.flatten(order="F")
        param_vector = np.concatenate([current_state, ref_colmajor, np.array([v_ref])])

        try:
            self.optimizer.set_value(self.params, param_vector)

            # MPC SOLVE TIME START
            _t_solve_start = time.perf_counter()

            # Solve optimization
            sol = self.optimizer.solve()

            # MPC SOLVE TIME END
            _t_solve_ms = (time.perf_counter() - _t_solve_start) * 1000
            print(f"[KIN_MPC_TIME] IPOPT Solve Time: {_t_solve_ms:.2f} ms")

            Uopt = sol.value(self.controls)
            Xopt = sol.value(self.states)

            # warm-start next step
            self._last_controls = Uopt
            self._last_states = Xopt

            # Extract controls
            delta = float(np.array(Uopt).reshape(-1)[0])
            delta = float(np.clip(delta, -self.max_steering, self.max_steering))

            print(f"[KIN_MPC] delta={delta:.4f} rad ({np.degrees(delta):.1f}°)")
            return delta

        except Exception as e:
            # Simple fallback: proportional on lateral error
            y_err = current_state[1] - reference_traj[0, 1]
            delta_fb = np.clip(-0.35 * y_err, -self.max_steering, self.max_steering)
            return float(delta_fb)