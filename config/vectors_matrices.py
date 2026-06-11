import numpy as np

T_CAM_TO_GRIPPER_GP7_ONHAND =  np.array([
        [0.0,     -1.0,   -0.3,   0.088],
        [1.0,      0.0,  0.013,  -0.035],
        [-0.01,   -0.3,   0.95,  -0.041],
        [0.0,      0.0,    0.0,     1.0]
    ])

T_CAM_TO_BASE_GP7_SPINDLE = np.array([
        [0.026,    0.0,     -1.0,    0.913],
        [1.0,      0.0,     0.026,  -0.175],
        [0.0,      -1.0,    0.0,    -0.336],
        [0.0,      0.0,     0.0,       1.0]
    ])

R_CAM_TO_BASE_GP7_SPINDLE = T_CAM_TO_BASE_GP7_SPINDLE[:3,:3]

#MARKER_TIP_IN_CAM = np.array([92.2, 21.9, 235.62]) #in mm
MARKER_TIP_IN_CAM = np.array([92.2, 23.3, 236.85])
MARKER_UNIT_VECTOR = np.array([0.0207,0.708, 0.617])

EXTEND_MARKER_TIP = MARKER_TIP_IN_CAM + MARKER_UNIT_VECTOR*2
ROBOT_X = np.array([1,0,0])
ROBOT_Z = np.array([0,0,1])