# -*- coding: UTF8 -*-
"""
    NAME
        ate_calc

    DESCRIPTION
        Module that implements methods to calculate
        the absolute trajectory error between two trajectories.

    METHODS
        compute_ate(gt_tst, pred_tst)
        compute_ate_horn(gt_tst, pred_tst)
        ate_xyz(alignment_error)
    EXAMPLES
        TODO: add
"""

import numpy as np
from .rpe_calc import get_statistics


def compute_ate(gt_tst, pred_tst):
    # get offset between first two poses
    offset = gt_tst[0] - pred_tst[0]

    #  convert both list to numpy array
    gt_tst_arr = np.array(gt_tst)
    pred_tst_arr = np.array(pred_tst)

    pred_tst_arr += offset

    # scaling factor
    scale = np.sum(gt_tst_arr * pred_tst_arr) / np.sum(pred_tst_arr ** 2)
    alignment_error = pred_tst_arr * scale - pred_tst_arr
    rmse = np.sqrt(np.sum(alignment_error ** 2))/len(gt_tst)

    return alignment_error


def compute_ate_horn(gt_tst, pred_tst):
    # make gt and pred columns
    gt_mat = np.ones(shape=(3, len(gt_tst)))
    pred_mat = np.ones(shape=(3, len(pred_tst)))

    for i, _ in enumerate(gt_tst):
        # mounts gt matrix
        temp_col = np.array(gt_tst[i]).reshape(3,1)
        gt_mat[:, i:i+1] = temp_col
        # mounts pred matrix
        temp_col = np.array(pred_tst[i]).reshape(3,1)
        pred_mat[:, i:i+1] = temp_col

    gt_mat_zero_centered = gt_mat - gt_mat.mean(1).reshape(3,1)
    pred_mat_zero_centered = pred_mat - pred_mat.mean(1).reshape(3,1)

    W = np.zeros((3, 3))
    for column in range(gt_mat.shape[1]):
        W += np.outer(gt_mat_zero_centered[:, column], pred_mat_zero_centered[:, column])

    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))

    if np.linalg.det(U) * np.linalg.det(Vh) < 0:
        S[2, 2] = -1

    rot = U * S * Vh
    trans = pred_mat.mean(1).reshape(3,1) - rot * gt_mat.mean(1).reshape(3,1)

    model_aligned = rot * gt_mat + trans
    alignment_error = model_aligned - pred_mat

    trans_error = np.sqrt(np.sum(np.multiply(alignment_error,alignment_error),0)).A[0]

    return alignment_error, trans_error


def ate_xyz(alignment_error):
    assert alignment_error.shape[0] == 3, f"Alignment vector must have 3 rows."

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
