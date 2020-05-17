# -*- coding: UTF8 -*-
"""
    NAME
        rpe_calc
    DESCRIPTION
        Module that implements methods to calculate
        the relative pose error between two trajectories.
    METHODS
        TODO: add
    EXAMPLES
        TODO: add
"""
import random
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

EPS = np.finfo(float).eps * 4.0


def translational_error(gt_tst, pred_tst):
    """
        Parameters
        ----------
            gt_tst : np.array
                The (x,y,z) of the ground truth.
            pred_tst : np.array
                 The (x,y,z) of the pred position.

        Returns
        -------
            np.float32
                The distance between two points.
    """
    return np.linalg.norm(gt_tst - pred_tst)


def rotational_error(gt_rot, pred_rot):
    """
        Parameters
        ----------
            gt_rot : np.array
                The (theta_x, theta_y, theta_z)
                of the ground truth.
            pred_rot : np.array
                 The (theta_x, theta_y, theta_z)
                 of the pred position.

        Returns
        -------
            np.float32
                The rotation between two points.
    """
    if gt_rot.shape == (3,):
        gt_rot = R.from_euler('xyz', gt_rot).as_dcm()

    if pred_rot.shape == (3,):
        pred_rot = R.from_euler('xyz', pred_rot).as_dcm()

    error_cos = 0.5 * (np.trace(pred_rot.dot(np.linalg.inv(gt_rot))) - 1.0)
    # Avoid invalid values due to numerical errors
    error_cos = min(1.0, max(-1.0, error_cos))
    error = math.acos(error_cos)

    return error
