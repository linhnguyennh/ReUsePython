import numpy as np
import os
import sys

from config.vectors_matrices import R_CAM_TO_BASE_GP7_SPINDLE, MARKER_UNIT_VECTOR

def make_3d_vector(point_start : np.ndarray, point_end : np.ndarray):
    return point_end - point_start

def rotate_vector_to_frame(vector_frame_B : np.ndarray, 
                           frame_B_to_A_rotation_matrix : np.ndarray = np.eye(3)):
    #uA = R_B_to_A x uB
    vector_frame_A = frame_B_to_A_rotation_matrix @ vector_frame_B
    return vector_frame_A

if __name__ == "__main__":
    vector_frame_B = make_3d_vector(np.array([1.0,2.0,3.0]),np.array([4.0,5.0,6.0]))
    print(rotate_vector_to_frame(vector_frame_B))

    marker_vector_in_base = rotate_vector_to_frame(MARKER_UNIT_VECTOR, R_CAM_TO_BASE_GP7_SPINDLE)
    print(marker_vector_in_base)