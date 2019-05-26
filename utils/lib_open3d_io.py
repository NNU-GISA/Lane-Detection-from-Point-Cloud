
import open3d
import numpy as np


def parse_data(xyza=None, xyz=None, rgb=None, a=None):
    '''Get valid cloud data'''

    # Check input
    c1 = xyza is not None
    c2 = (xyz is not None) and (rgb is not None)
    c3 = (xyz is not None) and (a is not None)
    assert any([c1, c2, c3]), "Invalid input"

    # Get xyz and rgb
    if xyza is not None:
        xyz = xyza[:, 0:3]
        a = xyza[:, 3]
    if rgb is None:
        N = xyz.shape[0]
        rgb = np.column_stack((a, np.zeros((N,)), 1-a))
    a = rgb[:, 0]
    xyza = np.column_stack((xyz, a))

    # return valid data
    return xyza, xyz, rgb, a


def write_cloud(filename, xyza=None, xyz=None, rgb=None, a=None):
    xyza, xyz, rgb, a = parse_data(xyza, xyz, rgb, a)
    cloud = form_cloud(xyza)
    open3d.write_point_cloud(filename, cloud)


def read_cloud(filename):
    cloud = open3d.read_point_cloud(filename)
    xyza = get_xyza(cloud)
    return xyza


def get_xyz_and_rgb(cloud):
    '''Get xyz and rgb from open3d cloud'''
    xyz = np.asarray(cloud.points)
    rgb = np.asarray(cloud.colors)
    return xyz, rgb


def get_xyza(cloud):
    '''Get xyz and alpha from open3d cloud'''
    '''Here alpha is stored in the first column of rgb'''
    xyz = np.asarray(cloud.points)
    rgb = np.asarray(cloud.colors)
    alpha = rgb[:, 0]
    xyza = np.column_stack((xyz, alpha))
    return xyza


def form_cloud(xyza=None, xyz=None, rgb=None, a=None):
    '''Form a open3d cloud from {xyz} and {rgb or alpha} data'''
    xyza, xyz, rgb, a = parse_data(xyza, xyz, rgb, a)
    open3d_cloud = open3d.PointCloud()
    open3d_cloud.points = open3d.utility.Vector3dVector(xyz)
    open3d_cloud.colors = open3d.utility.Vector3dVector(rgb)
    return open3d_cloud
