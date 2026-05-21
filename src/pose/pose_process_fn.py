import numpy as np
import scipy
from scipy.spatial.transform import Rotation
from math import pi
from dataclasses import dataclass

# @dataclass
# class ObjectPose:
#     pose_obj_to_cam : np.array = None
#     pose_obj_to_gripper : np.array = None
#     rotation_matrix : np.array = None
#     x_axis : np.array = get_x_axis(rotation_matrix)
#     y_axis : np.array = get_y_axis(rotation_matrix)
#     z_axis : np.array = get_z_axis(rotation_matrix)
#     position : np.array = None




def transform_pose(pose, T_B_to_A = np.eye(4)): # 4x4 pose with R and t
    transformed_pose = T_B_to_A @ pose #EXAMPLE: pose in cam frame (B) to pose in gripper frame (A) via T_C_G 
    return transformed_pose

def axis_angle(v_reference, v_target, symmetry=True):

    #Normalize
    v_reference = v_reference / np.linalg.norm(v_reference)
    v_target = v_target / np.linalg.norm(v_target)


    axis = np.cross(v_reference, v_target)
    if symmetry:
        angle = np.arccos(np.clip(abs(np.dot(v_reference,v_target)), -1.0, 1.0))
    else:
        angle = np.arccos(np.clip(np.dot(v_reference,v_target), -1.0, 1.0))
    #Missing: Check for v1 = v2 or v1 = -v2
    return axis, angle

def rotation_alignment_matrix(axis, angle): #Make rotation matrix from axis and angle
    rotvec = axis / np.linalg.norm(axis) * angle
    R_align = Rotation.from_rotvec(rotvec)
    return R_align

def euler_angles(R_align): #Convert rotation matrix to euler angles with intrinsic zyx sequence
    rz, ry, rx = R_align.as_euler(
        'zyx', #Intrinsic Z -> Y -> X from Yaskawa controller
        degrees=True
    )
    return rz, ry, rx


def align_axis(axis_reference =  np.array([1.0, 0.0, 0.0]),
                axis_target = np.array([1.0, 0.0, 0.0])): 
    
    #Perform all three steps in one function

    rz, ry, rx = None, None, None
    axis, angle = axis_angle(axis_reference, axis_target)
   
    R_align = rotation_alignment_matrix(axis, angle)

    rz, ry, rx = euler_angles(R_align)

    return rz, ry, rx

def pre_grasp_xyz(orig_coord, axis, offset_meter, gripper_depth): #Generate coordinate [offset] in meter amount away from an axis
    axis_normalized = axis / np.linalg.norm(axis)
    pre_grasp_coord = orig_coord - offset_meter * axis_normalized
    approach_vector = gripper_depth * axis_normalized
    return pre_grasp_coord, approach_vector

def is_pointing_away(axis1, axis2):
    axis1_norm = axis1 / np.linalg.norm(axis1)
    axis2_norm = axis2 / np.linalg.norm(axis2)

    dot = np.dot(axis1_norm, axis2_norm)
    if dot > 0:
        return True
    else:
        return False

def compare_dot_product(axis1, axis2, axis_reference):
    dot1 = np.dot(axis1, axis_reference)
    dot2 = np.dot(axis2, axis_reference)
    return (axis1, dot1) if abs(dot1) > abs(dot2) else (axis2, dot2)

def signed_angle(a, b, n):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    n = n / np.linalg.norm(n)

    cross = np.cross(a, b)
    sin_term = np.dot(cross, n)
    cos_term = np.dot(a, b)

    angle = np.arctan2(sin_term, cos_term)
    return np.degrees(angle)

def symmetric_angle(theta):
    theta = (theta + 180) % 360 - 180  # normalize to [-180, 180]

    if theta > 90:
        theta = 180 - theta
    elif theta < -90:
        theta = -180 - theta

    return theta

def get_rotation_matrix(T_matrix):
    return T_matrix[:3, :3]

def get_x_axis(R_matrix):
    return R_matrix[:,0]

def get_y_axis(R_matrix):
    return R_matrix[:,1]

def get_z_axis(R_matrix):
    return R_matrix[:,2]

def get_position(T_matrix):
    return T_matrix[:3,3]

if __name__ == "__main__":
    T_cam_to_gripper =  np.array([[0.0,     -1.0,   -0.3,   0.088],
                                [1.0,      0.0,  0.013,  -0.035],
                                [-0.01,   -0.3,   0.95,  -0.041],
                                [0.0,      0.0,    0.0,     1.0]])
    #STEP 0: orignal obj pose in camera frame
    pose_obj_to_cam = np.eye(4)
    #STEP 1: Convert obj pose in cam frame to gripper frame
    pose_obj_to_gripper = transform_pose(pose_obj_to_cam, T_cam_to_gripper)
    rotation_matrix = get_rotation_matrix(pose_obj_to_gripper)
    print(f"Transformed posed: \n {pose_obj_to_gripper}")
    print(f"Rotation matrix: \n {rotation_matrix}")
    print(f"X axis: {get_x_axis(rotation_matrix)}")
    print(f"Y axis: {get_y_axis(rotation_matrix)}")
    print(f"Z axis: {get_z_axis(rotation_matrix)}")

    #STEP 2: Calculate cross product and angle
    robot_z = [0,0,1]
    axis, angle = axis_angle(robot_z, [0.01,0.41,0.91])
    print(f"Axis: {axis}, Angle: {angle*180/pi}")
    rz,ry,rx = align_axis(np.array(robot_z), np.array([0.01,0.41,0.91]))
    print(f"RX: {rx:.2f}, RY: {ry:.2f}, RZ:{rz:.2f}")
    

    dx = np.dot(robot_z,[-1.04,0,0])
    dy = np.dot(robot_z,[0.01,0.41,0.91])
    dz = np.dot(robot_z,[-0.01,0.91,-0.4])

    print(f"dotX: {dx}, dotY: {dy}, dotZ: {dz}")
    #STEP 3: Calculate rotation matrix via Rodrigues
    #STEP 4: Calculate ZYX angle from rotation matrix using atan2
    #rz, ry, rx = align_axis()

    #FINAL: Return robot RX RY RZ angles
    
    

    