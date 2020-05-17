# -*- coding: UTF8 -*-
"""
    NAME
        visualize
    DESCRIPTION
        Module that implements methods to plot ground_truth trajectories
        and estimated for 6D poses.
    METHODS
        get_xy(pose)
            Function that extracts (x,y) positions from
            a list of 6D poses.
        get_seq_start(pose)
            Function that extracts the first (x,y)
            point from a list of 6D poses.
    EXAMPLES
        gt = np.load('04.npy')
        # create plot obj
        plt.clf()
        # get gt_poses
        gt_x, gt_y = get_xy(gt)
        # get sequence start
        x_start, y_start = get_seq_start(gt)
        # plot gt
        plt.scatter(x_start, y_start, label='Sequence Start', color='black')
        plt.plot(gt_x, gt_y, color='g', label='Ground Truth')
        # make the adjust for compute just translation
        # instead of absolute position
        plt.gca().set_aspect('equal', adjustable='datalim')
        # show plot
        plt.legend()
        plt.show()
"""
import numpy as np


def get_xy(pose):
    """
        Function that extracts (x,y) positions from
        a list of 6D poses.

        Parameters
        ----------
            pose : nd.array
                List of 6D poses.

        Returns
        -------
            x_pose : list
                List of x positions.
            y_pose : list
                List of y positions.
    """
    x_pose = [v for v in pose[:, 3]]
    y_pose = [v for v in pose[:, 5]]

    return x_pose, y_pose


def get_seq_start(pose):
    """
        Function that extracts the first (x,y)
        point from a list of 6D poses.

        Parameters
        ----------
            pose : nd.array
                List of 6D poses.

        Returns
        -------
            x_start : float
                Start x point.
            y_start : float
                Start y point
    """
    x_start = pose[0][3]
    y_start = pose[0][5]

    return x_start, y_start
