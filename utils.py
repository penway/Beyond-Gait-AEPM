import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid

used_joint_indexes = np.array([0,1,2,6,7,11,12,13,14,15,16,17,18,24,25,26]).astype(np.int64)
not_used_joint_indexes = np.array([3,4,5,8,9,10,19,20,21,22,23,27,28,29,30,31,32]).astype(np.int64)

def save_images(samples, path, nrow=8):
    images = []
    for sample in samples:
        res = get_pose_image(sample)
        # center crop to 400x400 from 640x480
        x_start = 640 // 2 - 400 // 2
        y_start = 480 // 2 - 400 // 2
        res = res[y_start:y_start+400, x_start:x_start+400, :]
        images.append(res)

    # turn list[ndarray] into ndarray
    images = np.array(images) # [B, H, W, C], 0-255

    # save image using torchvision
    images = torch.from_numpy(images).permute(0, 3, 1, 2).type(torch.uint8)
    images = images.float() / 255

    save_image(images, path, nrow=nrow)


def get_pose_image_batch(pose_info_list, frame_num=60):
    pose_xyzs = ninty_to_xyz(pose_info_list)
    if len(pose_xyzs.shape) == 4:
        pose_xyzs = pose_xyzs.reshape(-1, 32, 3)
        
    image_list = []
    for i in range(pose_xyzs.shape[0]):
        image_list.append(get_pose_image(pose_xyzs[i]))
    image_list = np.array(image_list)

    # center crop to 400x400 from 640x480
    x_start = 640 // 2 - 400 // 2
    y_start = 480 // 2 - 400 // 2
    image_list = image_list[:, y_start:y_start+400, x_start:x_start+400, :]
    images = torch.from_numpy(image_list).permute(0, 3, 1, 2).type(torch.uint8)
    images = images.float() / 255

    image_grid = make_grid(images, nrow=frame_num)
    return image_grid

def get_pose_image(data):
    data = data.cpu().numpy()
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-200, 200)
    ax.set_ylim3d(-200, 200)
    ax.set_zlim3d(-1000, 1000)
    ax.set_box_aspect([1, 1, 2000/400])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.plot(data[0:4,0], -data[0:4,2], data[0:4,1], color="blue")
    ax.plot(data[6:9,0], -data[6:9,2], data[6:9,1], color="blue")
    ax.plot([data[0,0], data[6,0]], [-data[0,2], -data[6,2]], [data[0,1], data[6,1]], color="blue")
    ax.plot(data[11:16, 0], -data[11:16, 2], data[11:16, 1], color="blue")
    ax.plot(data[16:20, 0], -data[16:20, 2], data[16:20, 1], color="blue")
    ax.plot(data[24:29, 0], -data[24:29, 2], data[24:29, 1], color="blue")

    # turn fig into numpy array
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return data

def sixteento32(pose_exp):
    # pad the not used joints with 0
    # pose_exp: [B, T, 16, 3] -> [B, T, 33, 3]
    b, t, _, _ = pose_exp.shape
    output = torch.zeros([b, t, 32, 3]).to(pose_exp.device)
    output[:, :, used_joint_indexes, :] = pose_exp

    return output



def sixteen_to_xyz(pose_exp):
    pose_exp = sixteento32(pose_exp)
    xyz = ninty_to_xyz(pose_exp)
    return xyz

def ninty_to_xyz(pose_exp):
    b, t, _, _ = pose_exp.shape
    
    # this is the original code, for 33 joints
    # pose_exp = torch.cat([torch.zeros(b, t, 1, 3).to(pose_exp.device), pose_exp], dim=2)
    # pose_exp = pose_exp.reshape(b*t, 33, 3)
    # pose_exp[:, :2] = 0
    # pose_exp = pose_exp[:, 1:, :].reshape(-1, 3)

    # this is the code for 32 
    pose_exp = pose_exp.reshape(b*t, 32, 3)
    pose_exp = pose_exp.reshape(-1, 3)

    pose_rot = expmap2rotmat_torch(pose_exp)
    pose_rot = pose_rot.reshape(b*t, 32, 3, 3)
    pose_xyz = rotmat2xyz_torch(pose_rot).reshape(b, t, 32, 3)
    return pose_xyz

def ninty_33_to_xyz(pose_exp):
    b, t, _, _ = pose_exp.shape
    
    # this is the original code, for 33 joints
    pose_exp = pose_exp.reshape(b*t, 33, 3)
    pose_exp[:, :2] = 0
    pose_exp = pose_exp[:, 1:, :].reshape(-1, 3)

    pose_rot = expmap2rotmat_torch(pose_exp)
    pose_rot = pose_rot.reshape(b*t, 32, 3, 3)
    pose_xyz = rotmat2xyz_torch(pose_rot).reshape(b, t, 32, 3)
    return pose_xyz

def sixteen_to_angle(pose_exp):
    pose_exp = sixteento32(pose_exp)
    pose_euler = ninty_to_angle(pose_exp)
    return pose_euler

def ninty_to_angle(pose_exp):

    b, t, _, _ = pose_exp.shape
    # pose_exp = pose_exp.reshape(b*t, 33, 3)
    # pose_exp[:, :2] = 0
    # pose_exp = pose_exp[:, 1:, :].reshape(-1, 3)

    pose_exp = pose_exp.reshape(b*t, 32, 3)
    pose_exp = pose_exp.reshape(-1, 3)
    
    pose_rot = expmap2rotmat_torch(pose_exp).reshape(b*t, 32, 3, 3)
    pose_euler = rotmat2euler_torch(pose_rot).reshape(b, t, 32, 3)
    return pose_euler

def ninty_33_to_angle(pose_exp):

    b, t, _, _ = pose_exp.shape
    pose_exp = pose_exp.reshape(b*t, 33, 3)
    pose_exp[:, :2] = 0
    pose_exp = pose_exp[:, 1:, :].reshape(-1, 3)
    
    pose_rot = expmap2rotmat_torch(pose_exp).reshape(b*t, 32, 3, 3)
    pose_euler = rotmat2euler_torch(pose_rot).reshape(b, t, 32, 3)
    return pose_euler


def ninty_to_xyz_99(pose_exp):
    # pose_exp: [B, T, 99] not [B, T, 33, 3], and output is [B, T, 32, 3]
    b, t, _ = pose_exp.shape
    pose_exp = pose_exp.reshape(b*t, 33, 3)
    pose_exp[:, :2] = 0
    pose_exp = pose_exp[:, 1:, :].reshape(-1, 3)
    pose_rot = expmap2rotmat_torch(pose_exp).reshape(b*t, 32, 3, 3)
    pose_xyz = rotmat2xyz_torch(pose_rot).reshape(b, t, 32, 3)
    return pose_xyz


# def ninty_to_xyz(hm1):
#     b, t, _, _ = hm1.shape
#     hm1 = hm1.reshape(b*t, -1)
#     num = hm1.shape[0]
#     # pose_info_list = np.zeros([num,32,3]) use torch.zeros instead
#     pose_info_list = torch.zeros([num,32,3])

#     for k in np.arange(0,num):
#         line = torch.zeros(33,3)
#         n = -1
#         for i in np.arange(0,99):
#             if i%3 == 0:
#                 n = n+1
#                 line[n][i%3]=hm1[k][i]
#             else:
#                 line[n][i%3]=hm1[k][i] # exponential map
#         pose_info = expmap2rotmat_torch(line).reshape(-1, 33, 3, 3)[:, 1:] # rotation matrix
#         pose_info = rotmat2xyz_torch(pose_info) # 3d coordinates
#         pose_info_list[k] = pose_info[0]

#     pose_info_list = pose_info_list.reshape(b, t, 32, 3)
#     return pose_info_list

def expmap2rotmat_torch(r):
    """
    Converts expmap matrix to rotation
    batch pytorch version ported from the corresponding method above
    :param r: N*3
    :return: N*3*3
    """
    theta = torch.norm(r, 2, 1)
    r0 = torch.div(r, theta.unsqueeze(1).repeat(1, 3) + 0.0000001)
    r1 = torch.zeros_like(r0).repeat(1, 3)
    r1[:, 1] = -r0[:, 2]
    r1[:, 2] = r0[:, 1]
    r1[:, 5] = -r0[:, 0]
    r1 = r1.view(-1, 3, 3)
    r1 = r1 - r1.transpose(1, 2)
    n = r1.data.shape[0]
    R = torch.eye(3, 3).repeat(n, 1, 1).float().to(r.device) + torch.mul(
        torch.sin(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3), r1) + torch.mul(
        (1 - torch.cos(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3)), torch.matmul(r1, r1))
    return R


def rotmat2xyz_torch(rotmat):
    """
    convert expmaps to joint locations
    :param rotmat: N*32*3*3
    :return: N*32*3
    """
    assert rotmat.shape[1] == 32
    parent, offset, rotInd, expmapInd = _some_variables()
    xyz = fkl_torch(rotmat, parent, offset, rotInd, expmapInd)
    return xyz

def rotmat2euler_torch(rotmat):
    """
    convert rotation matrix to euler angles
    :param rotmat: N*32*3*3
    :return: N*32*3
    """
    assert rotmat.shape[1] == 32
    # theta_y = R31
    # theta_x = atan2(R32, R33)
    # theta_z = atan2(R21, R11)
    theta_y = torch.asin(rotmat[:, :, 2, 0])
    theta_x = torch.atan2(rotmat[:, :, 2, 1], rotmat[:, :, 2, 2])
    theta_z = torch.atan2(rotmat[:, :, 1, 0], rotmat[:, :, 0, 0])
    return torch.stack([theta_x, theta_y, theta_z], dim=-1)


# def rotmat2euler_torch(rotmat):
#     """
#     Convert rotation matrix to Euler angles.
#     :param rotmat: Tensor of shape N*32*3*3 representing rotation matrices.
#     :return: Tensor of shape N*32*3 representing Euler angles.
#     """
#     assert rotmat.shape[1] == 32
#     # Extracting the elements for computing angles
#     R11 = rotmat[:, :, 0, 0]
#     R21 = rotmat[:, :, 1, 0]
#     R31 = rotmat[:, :, 2, 0]
#     R32 = rotmat[:, :, 2, 1]
#     R33 = rotmat[:, :, 2, 2]

#     # Threshold for detecting gimbal lock
#     gimbal_lock_threshold = 1e-6

#     # Calculating theta_y
#     theta_y = torch.asin(-R31)

#     # Check for gimbal lock
#     gimbal_lock = torch.abs(torch.cos(theta_y)) < gimbal_lock_threshold

#     # Initializing angles
#     theta_x = torch.zeros_like(theta_y)
#     theta_z = torch.zeros_like(theta_y)

#     # Handling the gimbal lock case
#     theta_z[gimbal_lock] = torch.atan2(-R21[gimbal_lock], R11[gimbal_lock])

#     # Handling the normal case
#     theta_x[~gimbal_lock] = torch.atan2(R32[~gimbal_lock], R33[~gimbal_lock])
#     theta_z[~gimbal_lock] = torch.atan2(R21[~gimbal_lock], R11[~gimbal_lock])

#     return torch.stack([theta_x, theta_y, theta_z], dim=-1)



def fkl_torch(rotmat, parent, offset, rotInd, expmapInd):
    """
    pytorch version of fkl.
    convert joint angles to joint locations
    batch pytorch version of the fkl() method above
    :param angles: N*99
    :param parent:
    :param offset:
    :param rotInd:
    :param expmapInd:
    :return: N*joint_n*3
    """
    n = rotmat.data.shape[0]
    j_n = offset.shape[0]
    p3d = torch.from_numpy(offset).float().to(rotmat.device).unsqueeze(0).repeat(n, 1, 1).clone()
    R = rotmat.view(n, j_n, 3, 3)
    for i in np.arange(1, j_n):
        if parent[i] > 0:
            R[:, i, :, :] = torch.matmul(R[:, i, :, :], R[:, parent[i], :, :]).clone()
            p3d[:, i, :] = torch.matmul(p3d[0, i, :], R[:, parent[i], :, :]) + p3d[:, parent[i], :]
    return p3d


def _some_variables():
    """
    borrowed from
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/forward_kinematics.py#L100
    We define some variables that are useful to run the kinematic tree
    Args
      None
    Returns
      parent: 32-long vector with parent-child relationships in the kinematic tree
      offset: 96-long vector with bone lenghts
      rotInd: 32-long list with indices into angles
      expmapInd: 32-long list with indices into expmap angles
    """

    parent = np.array([0, 1, 2, 3, 4, 5, 1, 7, 8, 9, 10, 1, 12, 13, 14, 15, 13,
                       17, 18, 19, 20, 21, 20, 23, 13, 25, 26, 27, 28, 29, 28, 31]) - 1

    offset = np.array(
        [0.000000, 0.000000, 0.000000, -132.948591, 0.000000, 0.000000, 0.000000, -442.894612, 0.000000, 0.000000,
         -454.206447, 0.000000, 0.000000, 0.000000, 162.767078, 0.000000, 0.000000, 74.999437, 132.948826, 0.000000,
         0.000000, 0.000000, -442.894413, 0.000000, 0.000000, -454.206590, 0.000000, 0.000000, 0.000000, 162.767426,
         0.000000, 0.000000, 74.999948, 0.000000, 0.100000, 0.000000, 0.000000, 233.383263, 0.000000, 0.000000,
         257.077681, 0.000000, 0.000000, 121.134938, 0.000000, 0.000000, 115.002227, 0.000000, 0.000000, 257.077681,
         0.000000, 0.000000, 151.034226, 0.000000, 0.000000, 278.882773, 0.000000, 0.000000, 251.733451, 0.000000,
         0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 99.999627, 0.000000, 100.000188, 0.000000, 0.000000,
         0.000000, 0.000000, 0.000000, 257.077681, 0.000000, 0.000000, 151.031437, 0.000000, 0.000000, 278.892924,
         0.000000, 0.000000, 251.728680, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 99.999888,
         0.000000, 137.499922, 0.000000, 0.000000, 0.000000, 0.000000])
    offset = offset.reshape(-1, 3)

    rotInd = [[5, 6, 4],
              [8, 9, 7],
              [11, 12, 10],
              [14, 15, 13],
              [17, 18, 16],
              [],
              [20, 21, 19],
              [23, 24, 22],
              [26, 27, 25],
              [29, 30, 28],
              [],
              [32, 33, 31],
              [35, 36, 34],
              [38, 39, 37],
              [41, 42, 40],
              [],
              [44, 45, 43],
              [47, 48, 46],
              [50, 51, 49],
              [53, 54, 52],
              [56, 57, 55],
              [],
              [59, 60, 58],
              [],
              [62, 63, 61],
              [65, 66, 64],
              [68, 69, 67],
              [71, 72, 70],
              [74, 75, 73],
              [],
              [77, 78, 76],
              []]

    expmapInd = np.split(np.arange(4, 100) - 1, 32)

    return parent, offset, rotInd, expmapInd
