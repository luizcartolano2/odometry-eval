# -*- coding: UTF8 -*-
"""
    NAME
        app
    DESCRIPTION
        TODO: add
    METHODS
        TODO: add
    EXAMPLES
        TODO: add
"""
import os
import sys
import pathlib
import numpy as np
import argparse
import matplotlib.pyplot as plt
from src.visualize import get_xy, get_seq_start
from src.rpe_calc import calculate_rpe_vector, get_statistics, calc_rpe_error
from src.ate_calc import ate_xyz, compute_ate_horn


def create_argparse():
    parser = argparse.ArgumentParser(prog='run',
                                     usage='%(prog)s [options] path',
                                     description='Run odometry metrics.')
    parser.add_argument('file_gt',
                        metavar='path',
                        type=str,
                        help='the path to ground truth file',
                        )

    parser.add_argument('file_pred',
                        metavar='path',
                        type=str,
                        help='the path to predict pose file',
                        )

    parser.add_argument('-v',
                        '--visualize',
                        help='visualize',
                        nargs='?',
                        const=1,
                        type=int)

    parser.add_argument('-ate',
                        help='absolute_error',
                        nargs='?',
                        const=1,
                        type=int)

    parser.add_argument('-rpe',
                        help='relative_error',
                        nargs='?',
                        const=1,
                        type=int)

    return parser


def read_file(filename):
    # read file according to extension
    extension = pathlib.Path(filename).suffix

    if extension == '.npy':
        file_pose = np.load(filename)
    elif extension == '.txt':
        with open(filename) as f_out:
            out = [l.split('\n')[0] for l in f_out.readlines()]
            for i, line in enumerate(out):
                out[i] = [float(v) for v in line.split(',')]
            file_pose = np.array(out)
    else:
        print('File extension not supported for gt_file.')
        sys.exit()

    return file_pose


if __name__ == '__main__':
    app_parser = create_argparse()
    args = app_parser.parse_args()

    # get args
    gt_path = args.file_gt
    pred_file = args.file_pred
    visualize_opt = args.visualize
    ate_opt = args.ate
    rpe_opt = args.rpe

    # first check if paths are valid
    if not os.path.isfile(gt_path):
        print('The ground truth path doesnt exist.')
        sys.exit()
    elif not os.path.isfile(pred_file):
        print('The predict path doesnt exist.')
        sys.exit()

    # convert file in np.arrays
    # of poses
    gt_poses = read_file(gt_path)
    pred_poses = read_file(pred_file)

    # now act according to the opt, if
    if visualize_opt is not None:
        # create plot obj
        plt.clf()

        # get gt_poses
        gt_x, gt_y = get_xy(gt_poses)
        # get predict x,y
        pred_x, pred_y = get_xy(pred_poses)

        # get sequence start
        x_start, y_start = get_seq_start(gt_poses)

        # plot gt
        plt.scatter(x_start, y_start, label='Sequence Start', color='black')
        plt.plot(gt_x, gt_y, color='g', label='Ground Truth')

        # plot predict
        plt.plot(pred_x, pred_y, color='r', label='Predict Pose')

        # make the adjust for compute just translation
        # instead of absolute position
        plt.gca().set_aspect('equal', adjustable='datalim')

        plt.legend()

        file_to_save = pathlib.Path(pred_file).stem
        plt.savefig(f'{file_to_save}_plot.png')

    if ate_opt is not None:
        # get translational attrs
        gt_tst = [v for v in gt_poses[:, 3:6]]
        pred_tst = [v for v in pred_poses[:, 3:]]

        alignment_error, trans_error = compute_ate_horn(gt_tst, pred_tst)
        statistics = ate_xyz(alignment_error)

        # write response to a .txt file
        file_to_save = pathlib.Path(pred_file).stem
        with open(f'{file_to_save}_ate.txt', 'w') as f:
            for k, v in statistics.items():
                f.write(f'{k}: \n\n')
                for key, value in v.items():
                    f.write(f'\t {key} >>> {value} \n\n')

    if rpe_opt is not None:
        # get translational attrs
        gt_tst = [v for v in gt_poses[:, 3:6]]
        pred_tst = [v for v in pred_poses[:, 3:]]

        # get rotational attrs
        gt_rot = [v for v in gt_poses[:, 0:3]]
        pred_rot = [v for v in pred_poses[:, 0:3]]

        # calculate rpe errors vector
        rpe_vector = calculate_rpe_vector(gt_tst, gt_rot, pred_tst, pred_rot)
        rpe_error_tst = calc_rpe_error(rpe_vector, error_type='translation_part')
        rpe_error_rot = calc_rpe_error(rpe_vector, error_type='rotation_part')

        # calculate errors statistics
        statistics_tst = get_statistics(rpe_error_tst)
        statistics_rot = get_statistics(rpe_error_rot)
        statistics = {
            'Translation': statistics_tst,
            'Rotation': statistics_rot,
        }

        # write response to a .txt file
        file_to_save = pathlib.Path(pred_file).stem
        with open(f'{file_to_save}_rpe.txt', 'w') as f:
            for k, v in statistics.items():
                f.write(f'{k}: \n\n')
                for key, value in v.items():
                    f.write(f'\t {key} >>> {value} \n\n')
