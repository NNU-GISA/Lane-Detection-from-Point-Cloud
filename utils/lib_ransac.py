
'''
This code is copied and modified from:
https://scipy-cookbook.readthedocs.io/items/RANSAC.html 
'''

import numpy as np
import scipy
import scipy.linalg
import time


def random_partition(n, N):
    # get n random indices out of [0, 1, 2, ..., N]
    indices = np.random.permutation(N)
    return indices[:n], indices[n:]


def ransac(
        data,  # P*N: p points, N feature dimension
        model,  # A class with methods:
                #   weight = fit(data)
                #   error = get_error(data, weights)
        n_pts_base,  # min points to fit model for 1st time
        n_pts_extra,  # min extra points to fit model for 2nd time
        max_iter,
        dist_thre,
        print_time=False,
        print_iter=True,
        debug=False):

    # Check input
    P, N = data.shape[0], data.shape[1]
    assert n_pts_base + \
        n_pts_extra < P, "Error: num points < (n_pts_base + n_pts_extra)"

    # Print param
    print("\n--------------------------------")
    print("Start RANSAC algorithm ...")
    print("Input: num points = {}, features dim = {}".format(P, N))
    print("Config: n_pts_base = {}, n_pts_extra = {}, dist_thre = {}".format(
        n_pts_base, n_pts_extra, dist_thre))
    print("")

    # Vars to record
    t0 = time.time()
    iter = 0
    best_weights = None
    best_err, best_num_pts = np.inf, 0
    best_inlier_idxs = None

    # Start iteration
    for iter in range(max_iter):
        if print_iter and (iter+1) % 2 == 0:
            print("{}, ".format(iter+1), end='')

        # Get may_be data, and fit may_be model
        maybe_idxs, test_idxs = random_partition(n_pts_base, P)
        maybe_data = data[maybe_idxs, :]
        maybe_weights = model.fit(maybe_data)

        # Remove bad data in may_be data
        maybe_err = model.get_error(maybe_data, maybe_weights)
        # select indices of rows with accepted points
        maybe_idxs = maybe_idxs[maybe_err < dist_thre]
        maybe_inliers = data[maybe_idxs, :]

        # Evaluate on test data
        test_points = data[test_idxs]
        test_err = model.get_error(test_points, maybe_weights)
        # select indices of rows with accepted points
        also_idxs = test_idxs[test_err < dist_thre]
        also_inliers = data[also_idxs, :]

        if debug and (iter == max_iter-1 or iter % 1 == 0):
            print("\n{}th iteration".format(iter))
            print('\ttest_err: mean = {}, min = {}, max = {}'.format(
                test_err.mean(), test_err.min(), test_err.max()))
            print('\ttotal fit points = {}'.format(len(also_inliers)))

        # Fit again
        if len(also_inliers) > n_pts_extra:
            better_idxs = np.concatenate((maybe_idxs, also_idxs))
            better_data = data[better_idxs, :]

            # Fit the model again
            idxs_to_fit = better_idxs.copy()
            if len(idxs_to_fit) > n_pts_base + n_pts_extra:
                # limit the number of points to fit to speed things up
                np.random.shuffle(idxs_to_fit)
                idxs_to_fit = idxs_to_fit[:n_pts_base + n_pts_extra]
            better_weights = model.fit(data[idxs_to_fit, :])

            # Remove bad data in may_be data
            better_err = model.get_error(better_data, better_weights)
            # select indices of rows with accepted points
            better_idxs = better_idxs[better_err < dist_thre]
            better_inliers = data[better_idxs, :]
            better_err = better_err[better_err < dist_thre]

            # criterias
            better_err_mean = np.mean(better_err)
            better_num_pts = sum(better_err < dist_thre)

            # Check criteria
            if 0:
                criteria = better_err_mean < best_err
            else:  # this one is better
                criteria = best_num_pts < better_num_pts

            if criteria:
                best_weights = better_weights
                best_err = better_err_mean
                best_num_pts = better_num_pts
                best_inlier_idxs = better_idxs
        continue

    if best_weights is None:
        raise ValueError("Didn't find any good model")

    # Print time
    if print_time:
        print("\nTime cost for RANSAC = {:.3f} seconds".format(
            time.time() - t0))
    print("--------------------------------\n")

    # Output
    return best_weights, best_inlier_idxs
