# -*- coding: UTF8 -*-
"""
    NAME
        ate_calc

    DESCRIPTION
        Module that implements methods to calculate
        the absolute trajectory error between two trajectories.

    METHODS
        compute_ate(gt_tst, pred_tst)
            Calculate the absolute trajectory
            error between two poses. Based on LearnerLee
            - KITTI_odometry_evaluation_tool repository.

        compute_ate_horn(gt_tst, pred_tst)
            Calculate the absolute trajectory error
            between two poses. Based on Horn align
            method. Explained by https://vision.in.tum.de/.

        ate_xyz(alignment_error)
            Calculate the statistics
            for separate axis.
    EXAMPLES
        # get translational attrs
        gt_tst = [v for v in gt_poses[:, 3:6]]
        pred_tst = [v for v in pred_poses[:, 3:]]

        alignment_error, trans_error = compute_ate_horn(gt_tst, pred_tst)
        statistics = ate_xyz(alignment_error)
"""

import numpy as np
from src.rpe_calc import get_statistics


def compute_ate(gt_tst, pred_tst):
    """
        Calculate the absolute trajectory
        error between two poses. Based on LearnerLee
        - KITTI_odometry_evaluation_tool repository.

        Parameters
        ----------
            gt_tst : list
                List of ground truth poses.
            pred_tst : list
                List of predict poses.

        Returns
        -------
            alignment_error : np.array (nx3)
                A matrix of errors by axis.
    """
    # get offset between first two poses
    offset = gt_tst[0] - pred_tst[0]

    #  convert both list to numpy array
    gt_tst_arr = np.array(gt_tst)
    pred_tst_arr = np.array(pred_tst)

    pred_tst_arr += offset

    # scaling factor
    scale = np.sum(gt_tst_arr * pred_tst_arr) / np.sum(pred_tst_arr ** 2)
    alignment_error = pred_tst_arr * scale - pred_tst_arr
    # rmse = np.sqrt(np.sum(alignment_error ** 2))/len(gt_tst)

    return alignment_error


def compute_ate_horn(gt_tst, pred_tst):
    """
        Calculate the absolute trajectory error
        between two poses. Based on Horn align
        method. Explained by https://vision.in.tum.de/.

        Parameters
        ----------
            gt_tst : list
                List of ground truth poses.
            pred_tst : list
                List of predict poses.

        Returns
        -------
            alignment_error : np.array (3xn)
                A matrix of errors by axis.
            trans_error : list
                The sum of error by row.
    """
    # make gt and pred columns
    gt_mat = np.ones(shape=(3, len(gt_tst)))
    pred_mat = np.ones(shape=(3, len(pred_tst)))

    for i, _ in enumerate(gt_tst):
        # mounts gt matrix
        temp_col = np.array(gt_tst[i]).reshape(3, 1)
        gt_mat[:, i:i+1] = temp_col
        # mounts pred matrix
        temp_col = np.array(pred_tst[i]).reshape(3, 1)
        pred_mat[:, i:i+1] = temp_col

    gt_mat_zero_centered = gt_mat - gt_mat.mean(1).reshape(3, 1)
    pred_mat_zero_centered = pred_mat - pred_mat.mean(1).reshape(3, 1)

    w_mat = np.zeros((3, 3))

    for column in range(gt_mat.shape[1]):
        w_mat += np.outer(gt_mat_zero_centered[:, column], pred_mat_zero_centered[:, column])

    u_mat, _, v_h = np.linalg.linalg.svd(w_mat.transpose())
    s_mat = np.matrix(np.identity(3))

    if np.linalg.det(u_mat) * np.linalg.det(v_h) < 0:
        s_mat[2, 2] = -1

    rot = u_mat * s_mat * v_h
    trans = pred_mat.mean(1).reshape(3, 1) - rot * gt_mat.mean(1).reshape(3, 1)

    model_aligned = rot * gt_mat + trans
    alignment_error = model_aligned - pred_mat

    trans_error = np.sqrt(np.sum(np.multiply(alignment_error, alignment_error), 0)).A[0]

    return alignment_error, trans_error


def ate_xyz(alignment_error):
    """
        Calculate the statistics
        for separate axis.

        Parameters
        ----------
            gt_tst : list
                List of ground truth poses.
            pred_tst : list
                List of predict poses.

        Returns
        -------
            alignment_error : np.array (3xn)
                A matrix of errors by axis.
            trans_error : list
                The sum of error by row.
    """
    assert alignment_error.shape[0] == 3, "Alignment vector must have 3 rows."

    x_error, y_error, z_error = [], [], []
    for column in range(alignment_error.shape[1]):
        # get temp row
        temp_row = alignment_error[:, column]

        # append individual errors
        x_error.append(temp_row[0].item())
        y_error.append(temp_row[1].item())
        z_error.append(temp_row[2].item())

    response_dict = {
        'x_ate': get_statistics(x_error),
        'y_ate': get_statistics(y_error),
        'z_ate': get_statistics(z_error),
    }

    return response_dict
