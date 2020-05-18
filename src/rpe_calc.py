# -*- coding: UTF8 -*-
"""
    NAME
        rpe_calc

    DESCRIPTION
        Module that implements methods to calculate
        the relative pose error between two trajectories.

    METHODS
        convert_pose_se3(pose_tst, pose_rot)
            Convert a rotation matrix (or euler angles)
            plus a translation vector into a 4x4 pose
            representation.

        relative_se3(pose_1, pose_2)
            Relative pose between two poses (drift).

        se3_inverse(pose)
            The inverse of a pose.

        calc_rpe_pair(Q_i, Q_i_delta, p_i, p_i_delta)
            The relative error between GT and Predict.

        so3_log(rot_matrix)
            Gets the rotation matrix from pose.

        calculate_rpe_vector(gt_tst, gt_rot, pred_tst, pred_rot)
            Gets a vector of relative errors for all poses.

        calc_rpe_error(error_vector, error_type='rotation_angle_deg')
            Calculate an specific error from relatives errors.

        get_statistics(rpe_vector)
            Statistics of a vector.

    EXAMPLES
        # calculate rpe errors vector
        rpe_vector = calculate_rpe_vector(gt_tst, gt_rot, pred_tst, pred_rot)
        rpe_error = calc_rpe_error(rpe_vector)

        # calculate errors statistics
        statistics = get_statistics(rpe_error)
"""
import numpy as np
from scipy.spatial.transform import Rotation as R


def convert_pose_se3(pose_tst, pose_rot):
    """
        Convert a rotation matrix (or euler angles)
        plus a translation vector into a 4x4 pose
        representation.

        Parameters
        ----------
            pose_tst : np.array
                The (x,y,z) of pose.
            pose_rot : np.array
                 The (theta_x, theta_y, theta_z) of pose.

        Returns
        -------
            np.array (4x4)
                The pose 4x4 matrix.
    """
    if pose_rot.shape == (3, 3):
        rot_mat = pose_rot
    else:
        rot_mat = R.from_euler('xyz', pose_rot).as_dcm()
    tst_vec = pose_tst

    se3 = np.eye(4)
    se3[:3, :3] = rot_mat
    se3[:3, 3] = tst_vec

    return se3


def relative_se3(pose_1, pose_2):
    """
        Relative pose between two poses (drift).

        Parameters
        ----------
            pose_1 : np.array
                The first pose.
            pose_2 : np.array
                 The second pose.

        Returns
        -------
            np.float32
                The relative transformation
                pose_1^{⁻1} * pose_2.
    """
    return np.dot(se3_inverse(pose_1), pose_2)


def se3_inverse(pose):
    """
        The inverse of a pose.

        Parameters
        ----------
            pose : np.array
                The pose.

        Returns
        -------
            np.float32
                The inverted pose.
    """
    r_inv = pose[:3, :3].transpose()
    t_inv = -r_inv.dot(pose[:3, 3])

    return convert_pose_se3(t_inv, r_inv)


def calc_rpe_pair(q_i, q_i_delta, p_i, p_i_delta):
    """
        The relative error between GT and Predict.

        Parameters
        ----------
            q_i : np.array
                The pose at time i.
            q_i_delta : np.array
                 The pose at time i + delta.
            p_i : np.array
                The predicted pose at time i.
            p_i_delta : np.array
                 The predicted pose at time i + delta.

        Returns
        -------
            np.float32
                The relative distance between two poses.
    """
    # get relative positions between
    # pose i and pose i + delta for the
    # Q (ground truth) and the P (predict)
    q_rel = relative_se3(q_i, q_i_delta)
    p_rel = relative_se3(p_i, p_i_delta)

    # get the relative error between then
    error = relative_se3(q_rel, p_rel)

    return error


def so3_log(rot_matrix):
    """
        Gets the rotation vector from
        rotation matrix.

        Parameters
        ----------
            rot_matrix : np.array
                The rotation matrix.

        Returns
        -------
            np.float32
                The error angle.
    """
    rotation_vector = R.from_dcm(rot_matrix).as_rotvec()
    angle = np.linalg.norm(rotation_vector)

    return angle


def calculate_rpe_vector(gt_tst, gt_rot, pred_tst, pred_rot):
    """
        Gets a vector of relative errors for all poses.

        Parameters
        ----------
            gt_tst : np.array
                The (x,y,z) of the ground truth.
            gt_rot : np.array
                The (theta_x, theta_y, theta_z) of the ground truth.
            pred_tst : np.array
                 The (x,y,z) of the predict pose.
            pred_rot : np.array
                The (theta_x, theta_y, theta_z) of the predict pose.

        Returns
        -------
            errors : list
                The list of relative errors.
    """
    errors = []
    for i in range(len(gt_tst) - 1):
        # ground truth
        gt_i = convert_pose_se3(gt_tst[i], gt_rot[i])
        gt_i_delta = convert_pose_se3(gt_tst[i+1], gt_rot[i+1])
        # predict
        pred_i = convert_pose_se3(pred_tst[i], pred_rot[i])
        pred_i_delta = convert_pose_se3(pred_tst[i+1], pred_rot[i+1])

        error_i = calc_rpe_pair(gt_i, gt_i_delta, pred_i, pred_i_delta)
        errors.append(error_i)
        # errors.append(abs(so3_log(error_i[:3, :3])) * 180 / np.pi)

    return errors


def calc_rpe_error(error_vector, error_type='rotation_angle_deg'):
    """
        Calculate an specific error from relatives errors.

        Parameters
        ----------
            error_vector : list
                List of relative errors.
            error_type : str
                Type of relative error to compute.

        Returns
        -------
            error : list
                The error asked by user.
    """
    if error_type == 'translation_part':
        error = [np.linalg.norm(error_i[:3, 3]) for error_i in error_vector]
    elif error_type == 'rotation_part':
        error = [np.linalg.norm(error_i[:3, :3] - np.eye(3)) for error_i in error_vector]
    elif error_type == 'rotation_angle_deg':
        error = [abs(so3_log(error_i[:3, :3])) * 180 / np.pi for error_i in error_vector]
    else:
        raise NotImplementedError

    return error


def get_statistics(rpe_vector):
    """
        Statistics of a vector.

        Parameters
        ----------
            rpe_vector : list
                List of errors.

        Returns
        -------
            dict
                Dict with statistics of a list.
    """
    return {
        'max': np.max(rpe_vector),
        'mean': np.mean(rpe_vector),
        'median': np.median(rpe_vector),
        'min': np.min(rpe_vector),
        'rmse': np.sqrt(np.mean(np.power(rpe_vector, 2))),
        'sse': np.sum(np.power(rpe_vector, 2)),
        'std': np.std(rpe_vector),
    }
