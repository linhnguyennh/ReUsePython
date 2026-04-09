import numpy as np
from scipy.spatial.transform import Rotation

def yaskawa_to_matrix(x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg):
    """
    Convert pendant readout to 4x4 homogeneous transform.
    """
    R = Rotation.from_euler('ZYX', [rz_deg, ry_deg, rx_deg], degrees=True).as_matrix()
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x_mm / 1000, y_mm / 1000, z_mm / 1000]  # to meters
    
    return T, R

T, R = yaskawa_to_matrix(838,162,700,180,-90,90)

#print(T)
print(R)