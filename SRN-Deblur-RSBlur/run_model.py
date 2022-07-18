import os
import argparse
import tensorflow as tf
# import models.model_gray as model
# import models.model_color as model
import models.model as model
import shutil
import numpy as np
import random


tf.set_random_seed(100)
np.random.seed(100)
random.seed(100)

def parse_args():
    parser = argparse.ArgumentParser(description='deblur arguments')
    parser.add_argument('--phase', type=str, default='test', help='determine whether train or test')
    parser.add_argument('--datalist', type=str, default='../datalist/RSBlur/RSBlur_real_train.txt', help='training datalist')
    parser.add_argument('--model', type=str, default='color', help='model type: [lstm | gray | color]')
    parser.add_argument('--batch_size', help='training batch size', type=int, default=16)
    parser.add_argument('--epoch', help='training epoch number', type=int, default=4000)
    parser.add_argument('--lr', type=float, default=1e-4, dest='learning_rate', help='initial learning rate')
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0', help='use gpu or cpu')
    parser.add_argument('--height', type=int, default=720,
                        help='height for the tensorflow placeholder, should be multiples of 16')
    parser.add_argument('--width', type=int, default=1280,
                        help='width for the tensorflow placeholder, should be multiple of 16 for 3 scales')
    parser.add_argument('--output_path', type=str, default='./testing_res',
                        help='output path for testing images')

    # new args
    parser.add_argument('--max_iteration', type=int, default=262000)
    parser.add_argument('--pre_trained', type=str, default='', help='pre_trained model path')
    parser.add_argument('--partial_load', type=int, default=0)
    parser.add_argument('--checkpoint_path', type=str, default='model', help='output path for testing images')
    parser.add_argument('--load_iteration', type=int, default=262000)


    # RSBlur args
    parser.add_argument('--dataset_aug', type=int, default=1)
    parser.add_argument('--no_mask', type=int, default=0)

    parser.add_argument('--sat_synthesis', type=str, default='None', choices=['sat_synthesis', 'None', 'oracle'])
    parser.add_argument('--sat_sacling_min', type=float, default=0.25)
    parser.add_argument('--sat_sacling_max', type=float, default=1.75)

    parser.add_argument('--noise_synthesis', type=str, default='None', choices=['poisson_RSBlur', 'poisson_gamma', 'gaussian', 'None'])

    parser.add_argument('--cam_params_RSBlur', type=int, default=0)
    parser.add_argument('--cam_params_RealBlur', type=int, default=0)

    parser.add_argument('--adopt_crf_realblur', type=int, default=0)

    parser.add_argument('--target_dataset', type=str, default='')
    parser.add_argument('--beta1', type=float, default=0)
    parser.add_argument('--beta2', type=float, default=0)

    args = parser.parse_args()
    return args


def main(_):
    args = parse_args()

    # set gpu/cpu mode
    if int(args.gpu_id) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # set up deblur models
    deblur = model.DEBLUR(args)
    if args.phase == 'test':
        deblur.test(args.height, args.width, args.output_path)
    elif args.phase == 'train':
        deblur.train()
    else:
        print('phase should be set to either test or train')


if __name__ == '__main__':
    tf.app.run()