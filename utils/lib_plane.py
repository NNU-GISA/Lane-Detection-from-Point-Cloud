
import numpy as np


def limit_points_number_for_PCA(points, max_points):
    ''' 
    Randomly select max_points from all points.
    Use this if your data points are more than 10000, otherwise SVD might fail.
    '''

    if points.shape[0] > max_points:
        rand_idx = np.random.permutation(points.shape[0])[:max_points]
        points = points[rand_idx, :]
    return points


def fit_plane_by_PCA(X, N=3):
    ''' Fit a hyperplane with dimension N by using PCA
    Input:
        X: data points, shape = (P, N)
    Output:
        w: coef of a hyperplane: w[0] + sigma(w[i] * x[i]) = 0
    '''

    # Check input X.shape = (P, 3)
    if X.shape[0] == N:
        X = X.T

    ''' 
    Compute PCA by svd algorithm:
        U, S, W = svd(Xc)
        if X=3*P, U[:, -1], last col is the plane norm
        if X=P*3, W[-1, :], last row is the plane norm
        Besides, S are the eigen values
    '''

    xm = np.mean(X, axis=0)  # 3
    Xc = X - xm[np.newaxis, :]
    U, S, W = np.linalg.svd(Xc)
    plane_normal = W[-1, :]  # 3

    '''
    Compute the bias:
        The fitted plane is this: w[1]*(x-xm)+w[2]*(x-ym)+w[3]*(x-zm)=0
        Change it back to the original:w[1]x+w[2]y+w[3]z+(-w[1]xm-w[2]ym-w[3]zm)=0
            --> w[0]=-w[1]xm-w[2]ym-w[3]zm
    '''
    w_0 = np.dot(xm, -plane_normal)
    w_1 = plane_normal
    w = np.concatenate(([w_0], w_1))
    return w


def fit_3D_line(x, y, z):
    '''
    Fit a line to {xi, yi, zi}, and compute the slope.
    Copied from: https://stackoverflow.com/questions/2298390/fitting-a-line-in-3d
    Input: 
        3 np.array
    Output:
        line direction, and a point on the line
    '''
    data = np.vstack((x, y, z)).T
    data_mean = data.mean(axis=0)
    uu, dd, vv = np.linalg.svd(data - data_mean)  # do SVD
    line_direction_vector = vv[0]  # vv[0] is the 1st principal vector
    return line_direction_vector, data_mean


class PlaneModel(object):
    ''' A class for fitting plane, by using "fit_plane_by_PCA" '''

    def __init__(self, feature_dimension=3):
        self.feature_dimension = feature_dimension
        # feature_dimension: 2 for line, 3 for plane

    def fit(self, X):
        '''
        Fit plane by PCA.
        Output:
            w: plane params, (bias, normal of the plane)
        '''
        # X.shape = (P, N)
        # w.shape    = (N+1, )
        N = self.feature_dimension

        # Check input X.shape = (P, 3)
        if X.shape[0] == N:
            X = X.T
        w = fit_plane_by_PCA(X, N)
        return w

    def get_error(self, data, w):
        point_to_plane_distance = (
            w[0] + np.dot(data, w[1:])) / np.linalg.norm(w[1:])
        return np.abs(point_to_plane_distance)


def plane_model(x, y, w=None, abc=None):
    ''' compute z based on x, y, and plane param'''

    if w is not None:  # w0 + w1*x + w2*y + w3*z = 0
        z = (-w[0] - w[1]*x - w[2]*y) / w[3]
    if abc is not None:
        a, b, c = abc
        z = a * x + b * y + c
    return z


def abc_to_w(abc):
    ''' Convert between different param reprensentations of a plane '''
    # z = ax + by + c ==>  c +  a*x +  b*y + -1*z = 0
    #                     w0 + w1*x + w2*y + w3*z = 0
    a, b, c = abc
    return [c, a, b, -1]


def w_to_abc(w):
    ''' Convert between different param reprensentations of a plane '''
    # z = ax + by + c ==>  c +  a*x +  b*y + -1*z = 0
    #                     w0 + w1*x + w2*y + w3*z = 0
    w0, w1, w2, w3 = w
    a, b, c = w1/(-w3), w2/(-w3), w0/(-w3)
    return [a, b, c]


def create_plane(
        weights_w=None, weights_abc=None,
        xy_range=(-5, 5, -5, 5), point_gap=1.0, noise=0,
        format="2D"):
    ''' Create scattered points of a plane.

    For line weights, choose either of "w" or "abc"    
    '''

    # Check input
    GAP = point_gap
    xl, xu, yl, yu = xy_range
    assert xl < xu and yl < yu

    # Create data
    xx, yy = np.mgrid[xl:xu:GAP, yl:yu:GAP]
    zz = plane_model(xx, yy, w=weights_w, abc=weights_abc)
    zz += np.random.random(zz.shape) * noise  # add noise

    # Output
    if format == "1D":
        x, y, z = map(np.ravel, [xx, yy, zz])  # matrix to array
        return x, y, z
    else:  # 2D
        return xx, yy, zz


'''
def calc_dist_to_line(x0,y0,x1,y1,x2,y2):
    a = (y2-y1)*x0-(x2-x1)*y0+x2*y1-y2*x1
    b = ((y2-y1)**2+(x2-x1)**2)**0.5
    return abs(a) / b

def calc_dist_to_line_segment(x, y, x1, y1, x2, y2):
    A, B, C, D = x - x1, y - y1, x2 - x1, y2 - y1
    dot = A * C + B * D
    len_sq = C * C + D * D
    param = -1
    if len_sq != 0:  # in case of 0 length line
        param = dot / len_sq
    if param < 0:
        xx = x1
        yy = y1
    elif param > 1:
        xx = x2
        yy = y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D
    intersect_point = (xx, yy)
    return ((x - xx)**2 + (y - yy)**2)**0.5, intersect_point
'''
