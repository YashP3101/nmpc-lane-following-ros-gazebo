#!/usr/bin/env python3

import numpy as np

def pcl2pxl(x_l, y_l, z_l):
    # Constants that do not change throughout the function
    fx = 476.53698  # focal_length in x
    fy = 476.57264  # focal_length in y
    cx = 399.86767  # center in x
    cy = 400.15544  # center in y
    tx = 1.8662     # translation in x
    ty = 0          # translation in y
    tz = -1.0981    # translation in z

    # Create translation vector t as a numpy array
    t = np.array([tx, ty, tz], dtype=np.float64)

    # Define rotation matrix R_lc
    R_lc = np.array([[0, -1, 0],
                    [0, 0, -1],
                    [1, 0, 0]], dtype=np.float64)
    
    # Create point P_l as a numpy array
    P_l = np.array([x_l, y_l, z_l], dtype=np.float64)

    # Compute the camera point P_c
    P_c = R_lc @ (P_l + t)

    # Calculate u and v in one step each
    u = (fx * P_c[0] / P_c[2]) + cx
    v = (fy * P_c[1] / P_c[2]) + cy

    # Round u and v and return them as a tuple
    return round(u), round(v)
