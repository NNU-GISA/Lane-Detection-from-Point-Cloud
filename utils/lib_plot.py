
import numpy as np
import open3d
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from .lib_geo_trans import transXYZ, rotx, roty, rotz, world2pixel
from .lib_cloud_proc import downsample
from matplotlib import gridspec


def plot_cloud_2d3d(xyza, figsize=(16, 8), title='', print_time=True):
    ''' Plot two figures for a point cloud: Left is 2d; Right is 3d '''
    t0 = time.time()
    fig = plt.figure(figsize=figsize)
    if 1:  # use "gridspec" to set display size
        gs = gridspec.GridSpec(2, 8)
        ax1 = fig.add_subplot(gs[:, 0:3])
        plot_cloud_2d(xyza, ax=ax1, title=title +
                      "\n(Number of points: {})".format(xyza.shape[0]))
        ax2 = fig.add_subplot(gs[:, 3:])
        plot_cloud_3d(xyza, ax=ax2)

    else:  # use "subplot" (However, this method cannot set display size)
        plt.subplot(1, 2, 1)
        plot_cloud_2d(xyza, ax=plt.gca(), title=title)
        plt.subplot(1, 2, 2)
        plot_cloud_3d(xyza, ax=plt.gca())

    fig.tight_layout()

    if print_time:
        print("Time cost of plotting 2D/3D point cloud = {:.2f} seconds".format(
            time.time() - t0))
    return ax1, ax2


def plot_cloud_2d(xyza, figsize=(8, 6), title='', ax=None):
    ''' Plot point cloud projected on x-y plane '''

    # Set figure
    if not ax:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    ax.set_aspect('equal')
    # xyza=downsample(xyza, voxel_size=0.1) # BE CAUSIOUS OF THIS !!!

    # Set color
    red = xyza[:, -1]
    green = np.zeros_like(red)
    blue = 1 - red
    color = np.column_stack((red, green, blue))

    # Set position
    x = xyza[:, 0]
    y = xyza[:, 1]

    # Plot
    plt.scatter(x, y, c=color, marker='.', linewidths=1)
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_title(title, fontsize=16)
    plt.axis('on')


def plot_cloud_3d(xyza, figsize=(12, 12), title='', ax=None):
    ''' Project 3d point cloud onto 2d image, and display'''

    # Create figure axes
    if not ax:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    num_points = xyza.shape[0]

    # Camera intrinsics
    w, h = 640, 480
    camera_intrinsics = np.array([
        [w, 0, w/2],
        [0, h, h/2],
        [0, 0,   1]
    ], dtype=np.float32)

    # Set view angle
    X, X, Z, ROTX, ROTY, ROTZ = 0, 0, 0, 0, 0, 0
    X, Y, Z = -20, -68, 238
    ROTZ = np.pi/2
    ROTY = -np.pi/2.8
    T_world_to_camera = transXYZ(x=X, y=Y, z=Z).dot(
        rotz(ROTZ)).dot(rotx(ROTX)).dot(roty(ROTY))
    T_cam_to_world = np.linalg.inv(T_world_to_camera)

    # Transform points' world positions to image pixel positions
    p_world = xyza[:, 0:3].T
    p_image = world2pixel(p_world, T_cam_to_world, camera_intrinsics)
    # to int, so it cloud be plot onto image
    p_image = np.round(p_image).astype(np.int)

    # Put each point onto image
    zeros, ones = np.zeros((h, w)), np.ones((h, w))
    color = np.zeros((h, w, 3))
    for i in range(num_points):  # iterate through all points
        x, y, a = p_image[0, i], p_image[1, i], xyza[i, -1]
        u, v = y, x  # flip direction to match the plt plot
        if w > u >= 0 and h > v >= 0:
            color[v][u][0] = max(color[v][u][0], a)
            color[v][u][2] = 1 - color[v][u][0]

    # Show
    ax.imshow(color)
    plt.axis('off')


# def plot_3d_cloud(cloud):

#     ''' Plot 3d points using Axes3D '''

#     if isinstance(cloud, open3d.PointCloud):
#         xyz = np.asarray(cloud.points)
#     else:
#         xyz = cloud[:, 0:3]

#     fig = plt.figure()
#     ax = Axes3D(fig)

#     x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
#     ax.scatter(x, y, z, marker='.', linewidth=1)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
