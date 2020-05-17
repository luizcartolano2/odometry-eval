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
import numpy as np

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
