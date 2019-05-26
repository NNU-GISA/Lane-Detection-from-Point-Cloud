
import open3d
import numpy as np
import collections
import scipy
from scipy import spatial
from .lib_open3d_io import form_cloud, get_xyza


def filter_cloud(xyza, nb_points=16, radius=1.0):
    ''' Filter cloud. Remove points with few neighbors.'''
    cloud = form_cloud(xyza=xyza)

    # Statistical oulier removal (no need to call this)
    if 0:
        cloud, inliers_ind = open3d.statistical_outlier_removal(
            cloud,
            nb_neighbors=20,
            std_ratio=2.0)

    # Radius oulier removal
    if 1:
        cloud, inliers_ind = open3d.radius_outlier_removal(
            cloud,
            nb_points=nb_points,
            radius=radius)

    xyza = get_xyza(cloud)
    return xyza


class KDTree_for_Points(object):
    def __init__(self, points):
        ''' Construct a kdtree using points 
        points.shape = (N, 3)
        N is number of points
        3 = (x, y, z)
        '''
        self.kdtree = spatial.KDTree(points.tolist())
        self.points = points

    def query(self, point, num_neighbors=1):
        dists, indices = self.kdtree.query(point, k=num_neighbors)
        # neighbor_points = [self.points[ind] for ind in indices]
        neighbor_points = self.points[indices, :]
        return neighbor_points


def find_plannar_points_by_kdtree(xyza, num_neighbors, max_height):
    ''' Find points whose neighbors have a △z <= max_height'''
    points = xyza[:, 0:3]
    kdtree = KDTree_for_Points(points)
    inliers = []
    for i, point in enumerate(points):
        if (i+1) % 2000 == 0:
            print("{}/{}".format(i+1, points.shape[0]), end=', ')
        neighbor_points = kdtree.query(point, num_neighbors=num_neighbors)
        list_z = neighbor_points[:, 2]
        if max(list_z) - min(list_z) <= max_height:
            inliers.append(i)
    return inliers


def find_plannar_points_by_grid(xyza, grid_size, max_height):
    ''' Find points whose grid have a △z <= max_height
    Be cautions, this function will round the x,y,z positions of a point to the grid size.
    '''

    scale = 1 / grid_size
    scale = int(scale) if scale > 1 else scale

    # Count height for each grid_size
    d = collections.defaultdict(list)
    xy = np.round(xyza[:, 0:2] * scale)
    for idx in range(xyza.shape[0]):
        x, y = xy[idx, :]
        z = xyza[idx, 2]
        d[(x, y)].append((idx, z))

    # Filter the grid. Remove those with large different in z axis
    inliers = []
    for xy, list_of_idx_and_z in d.items():
        list_idx, list_z = zip(*list_of_idx_and_z)
        if max(list_z) - min(list_z) <= max_height:
            inliers.extend(list_idx)

    return inliers


def downsample(xyza, voxel_size=0.01, option='max_alpha'):
    '''
    Downsample the cloud, while retaining the largest local alpha (intensity) value
    Input:
        voxel_size: (1/voxel_size) should be an integer
    '''

    assert option in ['max_alpha', 'mean_alpha']

    scale = 1 / voxel_size
    scale = int(scale) if scale > 1 else scale

    xyz = np.round(xyza[:, 0:3]*scale)
    alpha = xyza[:, 3]

    # Get points in each voxel and their alphas
    d = collections.defaultdict(list)  # dict: grid pos --> list of alphas
    for x, y, z, a in np.column_stack((xyz, alpha)):
        pos = (x, y, z)
        d[pos].append(a)

    # Compute alpha of each voxel
    def max_(arr):
        return max(arr)

    def mean_(arr):
        return 1.0 * sum(arr) / len(arr)

    if option == "max_alpha":
        dict_pos_to_alpha = {pos: max_(alphas) for (pos, alphas) in d.items()}
    elif option == "mean_alpha":
        dict_pos_to_alpha = {pos: mean_(alphas) for (pos, alphas) in d.items()}

    # Get downsampled points from dict
    new_xyz = 1.0 * np.array(list(dict_pos_to_alpha.keys())) / scale
    new_a = np.array(list(dict_pos_to_alpha.values()))
    new_xyza = np.column_stack((new_xyz, new_a))
    return new_xyza


def get_xy_from_latlon(lats, lons):
    ''' Extablish a Cartesian coordinate at the mean position of (lats, lons)
        and convert (lats, lons) to this local Cartesian coordinate as (xs, ys)
    '''
    R = 6371*1000
    lat_mean, lon_mean = np.mean(lats)/180*np.pi, np.mean(lons)/180*np.pi
    alpha, beta = lon_mean, lat_mean

    def sphere_to_world_coordinate(lat, lon):
        alpha, beta = lon, lat
        return R * np.array([
            np.cos(beta)*np.cos(alpha),
            np.cos(beta)*np.sin(alpha),
            np.sin(beta)
        ])

    OL = sphere_to_world_coordinate(lat_mean, lon_mean)
    XL = np.array([-np.sin(alpha), np.cos(alpha), 0])
    YL = np.cross(OL, XL) / np.linalg.norm(OL)

    lats = lats/180*np.pi
    lons = lons/180*np.pi
    OPs = sphere_to_world_coordinate(lats, lons)
    cartisian_coordinates = np.vstack((XL, YL))
    x_y_pos = np.dot(cartisian_coordinates, OPs - OL[:, np.newaxis]).T

    # Store x, y positions into pandas
    xs = x_y_pos[:, 0]
    ys = x_y_pos[:, 1]
    return xs, ys
