from __future__ import print_function
import os
import time
import random
import datetime

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from datetime import datetime
from util.util import *
from util.BasicConvLSTMCell import *
import math
import cv2
import collections
from ISP.ISP_implement_tf_malvar import ISP, rgb2lin_np, lin2rgb_np, rgb2lin_tf, lin2rgb_tf


try:
    from scipy.misc import imread
    from scipy.misc import imsave
except:
    from imageio import imread
    from imageio import imsave

class DEBLUR(object):
    def __init__(self, args):
        self.args = args
        self.n_levels = 3
        self.scale = 0.5
        self.chns = 3 if self.args.model == 'color' else 1  # input / output channels

        # if args.phase == 'train':
        self.crop_size = 256

        self.data_list = []
        data_list1 = args.datalist.split(',')[0]
        data_list1 = open(data_list1, 'rt').read().splitlines()
        data_list1 = list(map(lambda x: x.strip().split(' '), data_list1))
        self.data_list += data_list1

        print('training sample number : ', len(self.data_list))
        random.shuffle(self.data_list)


        self.data_list2 = []
        if len(args.datalist.split(',')) >= 2:
            data_list2 = args.datalist.split(',')[1]
            data_list2 = open(data_list2, 'rt').read().splitlines()
            data_list2 = list(map(lambda x: x.strip().split(' '), data_list2))
            self.data_list2 += data_list2
        random.shuffle(self.data_list2)

        self.train_dir = os.path.join('./checkpoints', args.checkpoint_path)
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.max_steps = self.args.max_iteration #524000
        self.learning_rate = args.learning_rate

        self.isp = ISP()
        if self.args.target_dataset == 'RSBlur':
            self.isp.beta1 = 0.0001
            self.isp.beta2 = 9.1504e-04

        if self.args.target_dataset == 'RealBlur':

            self.isp.beta1 = 8.8915e-05
            self.isp.beta2 = 2.9430e-05

        if self.args.target_dataset == 'sidd_gp':
            self.isp.beta1 = 0.000228107086
            self.isp.beta2 = 0.0000021247339

        if self.args.beta1 > 0:
            self.isp.beta1 = self.args.beta1
            self.isp.beta2 = self.args.beta2

    def input_producer(self, batch_size=10, data_list=None):
        def read_data(data_queue):
            img_a = tf.image.decode_image(tf.read_file(tf.string_join(['./dataset/', data_queue[0]])),
                                          channels=3)
            img_b = tf.image.decode_image(tf.read_file(tf.string_join(['./dataset/', data_queue[1]])),
                                          channels=3)

            # saturation mask
            if self.args.no_mask:
                mask_img = tf.ones_like(img_a)
            else:
                mask_name = '_mask_100/'
                mask_path = tf.strings.regex_replace(data_queue[0], "_img/", mask_name)
                mask_img = tf.image.decode_image(tf.read_file(tf.string_join(['./dataset/', mask_path])), channels=3)

            # for less artifacts of demosaic methods
            if 'poisson' in self.args.noise_synthesis:
                new_img_list = preprocessing([img_a, img_b, mask_img], self.crop_size+8)
            else:
                new_img_list = preprocessing([img_a, img_b, mask_img], self.crop_size)

            img_a = new_img_list[0]
            img_b = new_img_list[1]
            mask_img = new_img_list[2]

            # saturation synthesis
            if self.args.sat_synthesis == 'sat_synthesis':
                random_scaling = tf.random.uniform([1], minval=self.args.sat_sacling_min, maxval=self.args.sat_sacling_max, dtype=tf.float32, seed=None, name=None)
                img_a = rgb2lin_tf(img_a)
                img_a = img_a + (mask_img * random_scaling)
                img_a = lin2rgb_tf(img_a)
                img_a = tf.clip_by_value(img_a, 0, 1)
            elif self.args.sat_synthesis == 'oracle':
                pass
            elif self.args.sat_synthesis == 'None':
                pass
            else:
                raise RuntimeError('plz check sat_synthesis params')

            xyz2cam, wb = None, None

            # find camera parameters of a image
            if self.args.cam_params_RSBlur:
                def random_camera_params_known(data_path):
                    inp_path = str(data_path)
                    key = inp_path.split('/')[-4]

                    M_xyz2cam = self.isp.xyz2cam_list[key]
                    M_xyz2cam = np.transpose(M_xyz2cam)

                    fr_now = 1 / self.isp.wbs[key][0]
                    fb_now = 1 / self.isp.wbs[key][2]

                    wb = np.array([fr_now, fb_now])

                    return M_xyz2cam.astype('float32'), wb.astype('float32')

                xyz2cam, wb = tf.numpy_function(func=random_camera_params_known, inp=[data_queue[1]], Tout=[tf.float32, tf.float32])
                xyz2cam.set_shape((3, 3))
                wb.set_shape((2))

            if self.args.cam_params_RealBlur:

                # read all wb values of realblur
                with open('ISP/mat_collections/realblur_iso_wb_train.txt', 'rt') as f:
                    wb_list = f.readlines()

                wb_list = [wb_path.strip().split(' ') for wb_path in wb_list]
                wb_list = [[float(wb_path[2]), float(wb_path[3])] for wb_path in wb_list]
                wb_np = 1/np.array(wb_list)
                wb_tf = tf.constant(wb_np.astype('float32'))

                # random sampling
                random_index = tf.random.uniform([1], minval=0, maxval=(wb_np.shape[0]-1), dtype=tf.int64, seed=None, name=None)
                wb = wb_tf[random_index[0],:]

                # ccm matrix from libraw
                xyz2cam = tf.constant(self.isp.xyz2cam_realblur.astype('float32'))

            if xyz2cam is None and wb is None:
                xyz2cam = tf.zeros([3,3])
                wb = tf.zeros([2])

            return img_a, img_b, mask_img, xyz2cam, wb

        def preprocessing(imgs, crop_size):
            imgs = [tf.cast(img, tf.float32) / 255.0 for img in imgs]
            if self.args.model != 'color':
                imgs = [tf.image.rgb_to_grayscale(img) for img in imgs]
            out = tf.random_crop(tf.stack(imgs, axis=0), [len(imgs), crop_size, crop_size, self.chns])

            if self.args.dataset_aug:
                random_int = tf.random_uniform(shape=[], minval=0, maxval=3, dtype=tf.int32)
                out = tf.image.rot90(out, k=random_int)

            out = tf.unstack(out, axis=0)

            if self.args.dataset_aug:
                do_flip = tf.random_uniform([]) > 0.5
                out = [tf.cond(do_flip, lambda: tf.image.flip_left_right(img), lambda: img) for img in out]

            if self.args.dataset_aug:
                do_flip = tf.random_uniform([]) > 0.5
                out = [tf.cond(do_flip, lambda: tf.image.flip_up_down(img), lambda: img) for img in out]

            return out

        with tf.variable_scope('input'):
            List_all = tf.convert_to_tensor(data_list, dtype=tf.string)
            gt_list = List_all[:, 0]
            in_list = List_all[:, 1]

            data_queue = tf.train.slice_input_producer([in_list, gt_list], capacity=20)
            image_in, image_gt, sat_mask, image_xyz2cam, image_wb = read_data(data_queue)
            batch_in, batch_gt, batch_sat_mask, batch_xyz2cam, batch_wb = tf.train.batch(
                [image_in, image_gt, sat_mask, image_xyz2cam, image_wb], batch_size=batch_size,
                num_threads=8, capacity=20)

        return batch_in, batch_gt, batch_sat_mask, batch_xyz2cam, batch_wb

    def generator(self, inputs, reuse=False, scope='g_net'):
        n, h, w, c = inputs.get_shape().as_list()

        if self.args.model == 'lstm':
            with tf.variable_scope('LSTM'):
                cell = BasicConvLSTMCell([h / 4, w / 4], [3, 3], 128)
                rnn_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        x_unwrap = []
        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                activation_fn=tf.nn.relu, padding='SAME', normalizer_fn=None,
                                weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                biases_initializer=tf.constant_initializer(0.0)):

                inp_pred = inputs
                for i in xrange(self.n_levels):
                    scale = self.scale ** (self.n_levels - i - 1)
                    hi = int(round(h * scale))
                    wi = int(round(w * scale))
                    inp_blur = tf.image.resize_images(inputs, [hi, wi], method=0)
                    inp_pred = tf.stop_gradient(tf.image.resize_images(inp_pred, [hi, wi], method=0))
                    inp_all = tf.concat([inp_blur, inp_pred], axis=3, name='inp')
                    if self.args.model == 'lstm':
                        rnn_state = tf.image.resize_images(rnn_state, [hi // 4, wi // 4], method=0)

                    # encoder
                    conv1_1 = slim.conv2d(inp_all, 32, [5, 5], scope='enc1_1')
                    conv1_2 = ResnetBlock(conv1_1, 32, 5, scope='enc1_2')
                    conv1_3 = ResnetBlock(conv1_2, 32, 5, scope='enc1_3')
                    conv1_4 = ResnetBlock(conv1_3, 32, 5, scope='enc1_4')
                    conv2_1 = slim.conv2d(conv1_4, 64, [5, 5], stride=2, scope='enc2_1')
                    conv2_2 = ResnetBlock(conv2_1, 64, 5, scope='enc2_2')
                    conv2_3 = ResnetBlock(conv2_2, 64, 5, scope='enc2_3')
                    conv2_4 = ResnetBlock(conv2_3, 64, 5, scope='enc2_4')
                    conv3_1 = slim.conv2d(conv2_4, 128, [5, 5], stride=2, scope='enc3_1')
                    conv3_2 = ResnetBlock(conv3_1, 128, 5, scope='enc3_2')
                    conv3_3 = ResnetBlock(conv3_2, 128, 5, scope='enc3_3')
                    conv3_4 = ResnetBlock(conv3_3, 128, 5, scope='enc3_4')

                    if self.args.model == 'lstm':
                        deconv3_4, rnn_state = cell(conv3_4, rnn_state)
                    else:
                        deconv3_4 = conv3_4

                    # decoder
                    deconv3_3 = ResnetBlock(deconv3_4, 128, 5, scope='dec3_3')
                    deconv3_2 = ResnetBlock(deconv3_3, 128, 5, scope='dec3_2')
                    deconv3_1 = ResnetBlock(deconv3_2, 128, 5, scope='dec3_1')
                    deconv2_4 = slim.conv2d_transpose(deconv3_1, 64, [4, 4], stride=2, scope='dec2_4')
                    cat2 = deconv2_4 + conv2_4
                    deconv2_3 = ResnetBlock(cat2, 64, 5, scope='dec2_3')
                    deconv2_2 = ResnetBlock(deconv2_3, 64, 5, scope='dec2_2')
                    deconv2_1 = ResnetBlock(deconv2_2, 64, 5, scope='dec2_1')
                    deconv1_4 = slim.conv2d_transpose(deconv2_1, 32, [4, 4], stride=2, scope='dec1_4')
                    cat1 = deconv1_4 + conv1_4
                    deconv1_3 = ResnetBlock(cat1, 32, 5, scope='dec1_3')
                    deconv1_2 = ResnetBlock(deconv1_3, 32, 5, scope='dec1_2')
                    deconv1_1 = ResnetBlock(deconv1_2, 32, 5, scope='dec1_1')
                    inp_pred = slim.conv2d(deconv1_1, self.chns, [5, 5], activation_fn=None, scope='dec1_0')

                    if i >= 0:
                        x_unwrap.append(inp_pred)
                    if i == 0:
                        tf.get_variable_scope().reuse_variables()

            return x_unwrap

    def build_model(self):

        batch_size = self.batch_size
        img_in_before, img_gt, img_mask, img_xyz2cam, img_wb = self.input_producer(batch_size, self.data_list)

        if self.args.noise_synthesis=='gaussian':
            random_std = 0.011211179037436594

            min_noise = random_std * 0.5
            noise_std = min_noise + tf.random.uniform([batch_size], 0, 1) * (random_std * 1.5 - min_noise)
            noise_std = tf.reshape(noise_std, [batch_size, 1, 1, 1])

            img_in = img_in_before + (tf.random_normal(shape=tf.shape(img_in_before), mean=0.0, stddev=1, dtype=tf.float32) * noise_std)
        elif self.args.noise_synthesis == 'poisson_RSBlur':
            img_in = self.isp.poisson_RSBlur(img_in_before, img_xyz2cam, img_wb[:, 0], img_wb[:, 1])
        elif self.args.noise_synthesis == 'poisson_gamma':
            img_in = self.isp.poisson_gamma(img_in_before, img_xyz2cam, img_wb[:,0], img_wb[:,1])
        elif self.args.noise_synthesis == 'None':
            img_in = img_in_before
        else:
            raise RuntimeError('plz check noise_synthesis params')

        img_in = tf.clip_by_value(img_in, 0, 1)

        if self.args.sat_synthesis != 'None':
            sat_mask = tf.math.greater_equal(img_in_before , 1.0)
            non_sat_mask = tf.math.logical_not(sat_mask)

            sat_mask = tf.cast(sat_mask, tf.float32)
            non_sat_mask = tf.cast(non_sat_mask, tf.float32)

            img_in = img_in * non_sat_mask + img_in_before * sat_mask

        if self.args.adopt_crf_realblur:
            img_in = rgb2lin_tf(img_in)
            img_in = self.isp.lin2rgb_realblur(img_in)
            img_gt = rgb2lin_tf(img_gt)
            img_gt = self.isp.lin2rgb_realblur(img_gt)

        # for less artifacts of demosaic
        if 'poisson' in self.args.noise_synthesis:
            img_in = img_in[:, 4:-4, 4:-4, :]
            img_in_before = img_in_before[:, 4:-4, 4:-4, :]
            img_gt = img_gt[:, 4:-4, 4:-4, :]
            img_mask = img_mask[:, 4:-4, 4:-4, :]

        print('img_in, img_gt', img_in.get_shape(), img_gt.get_shape())
        tf.summary.image('img_in_before', im2uint8(img_in_before))
        tf.summary.image('img_in', im2uint8(img_in))
        tf.summary.image('img_gt', im2uint8(img_gt))
        tf.summary.image('img_mask', im2uint8(img_mask/(tf.reduce_max(img_mask, axis=[1,2], keepdims=True)+0.000001)))

        # generator
        x_unwrap = self.generator(img_in, reuse=False, scope='g_net')
        # calculate multi-scale loss
        self.loss_total = 0
        for i in xrange(self.n_levels):
            _, hi, wi, _ = x_unwrap[i].get_shape().as_list()
            gt_i = tf.image.resize_images(img_gt, [hi, wi], method=0)
            loss = tf.reduce_mean((gt_i - x_unwrap[i]) ** 2)
            self.loss_total += loss

            tf.summary.image('out_' + str(i), im2uint8(x_unwrap[i]))
            tf.summary.scalar('loss_' + str(i), loss)

        # losses
        tf.summary.scalar('loss_total', self.loss_total)

        # training vars
        all_vars = tf.trainable_variables()
        self.all_vars = all_vars
        self.g_vars = [var for var in all_vars if 'g_net' in var.name]
        self.lstm_vars = [var for var in all_vars if 'LSTM' in var.name]
        #for var in all_vars:
        #    print(var.name)

    def train(self):
        def get_optimizer(loss, global_step=None, var_list=None, is_gradient_clip=False):
            train_op = tf.train.AdamOptimizer(self.lr)
            if is_gradient_clip:
                grads_and_vars = train_op.compute_gradients(loss, var_list=var_list)
                unchanged_gvs = [(grad, var) for grad, var in grads_and_vars if not 'LSTM' in var.name]
                rnn_grad = [grad for grad, var in grads_and_vars if 'LSTM' in var.name]
                rnn_var = [var for grad, var in grads_and_vars if 'LSTM' in var.name]
                capped_grad, _ = tf.clip_by_global_norm(rnn_grad, clip_norm=3)
                capped_gvs = list(zip(capped_grad, rnn_var))
                train_op = train_op.apply_gradients(grads_and_vars=capped_gvs + unchanged_gvs, global_step=global_step)
            else:
                train_op = train_op.minimize(loss, global_step, var_list)
            return train_op

        global_step = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False)
        self.global_step = global_step

        # build model
        self.build_model()

        # learning rate decay
        self.lr = tf.train.polynomial_decay(self.learning_rate, global_step, self.max_steps, end_learning_rate=0.0, power=0.3)


        tf.summary.scalar('learning_rate', self.lr)

        # training operators
        train_gnet = get_optimizer(self.loss_total, global_step, self.all_vars)

        # session and thread
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess = sess

        if self.args.pre_trained == '':
            self.saver = tf.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=1)
            sess.run(tf.global_variables_initializer())
        elif self.args.pre_trained != '':
            self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=50, keep_checkpoint_every_n_hours=1)
            sess.run(tf.global_variables_initializer())
            self.load(sess, self.args.pre_trained, self.args.load_iteration)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # training summary
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph, flush_secs=30)

        for step in xrange(sess.run(global_step), self.max_steps + 1):

            start_time = time.time()

            # update G network
            _, loss_total_val = sess.run([train_gnet, self.loss_total])

            duration = time.time() - start_time
            # print loss_value
            assert not np.isnan(loss_total_val), 'Model diverged with loss = NaN'

            if step % 5 == 0:
                num_examples_per_step = self.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = (%.5f; %.5f, %.5f)(%.1f data/s; %.3f s/bch)')
                print(format_str % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), step, loss_total_val, 0.0,
                                    0.0, examples_per_sec, sec_per_batch))

            if step % 500 == 0:
                # summary_str = sess.run(summary_op, feed_dict={inputs:batch_input, gt:batch_gt})
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step=step)

            # Save the model checkpoint periodically.
            if step % 20000 == 0 or step == self.max_steps:
                checkpoint_path = os.path.join(self.train_dir, 'checkpoints')
                self.save(sess, checkpoint_path, step)

    def save(self, sess, checkpoint_dir, step):
        model_name = "deblur.model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, sess, checkpoint_dir, step=None):
        print(" [*] Reading checkpoints...")
        model_name = "deblur.model"
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if step is not None:
            ckpt_name = model_name + '-' + str(step)
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Reading intermediate checkpoints... Success")
            return str(step)
        elif ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            ckpt_iter = ckpt_name.split('-')[1]
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Reading updated checkpoints... Success")
            return ckpt_iter
        else:
            print(" [*] Reading checkpoints... ERROR")
            return False


    def test(self, height, width, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_path = os.path.join(output_path, self.args.checkpoint_path)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        H, W = height, width
        inp_chns = 3 if self.args.model == 'color' else 1
        self.batch_size = 1 if self.args.model == 'color' else 3

        network_dict = collections.defaultdict(list)
        inputs = tf.placeholder(shape=[self.batch_size, H, W, inp_chns], dtype=tf.float32)
        outputs = self.generator(inputs, reuse=False)

        network_dict['%dx%d' % (H,W)] = [inputs, outputs]

        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

        self.saver = tf.train.Saver()
        self.load(sess, self.train_dir, self.args.load_iteration)

        data_list = open(self.args.datalist, 'rt').read().splitlines()
        data_list = list(map(lambda x: x.strip().split(' '), data_list))

        total_psnr = 0
        total_psnr_input = 0
        total_mse = 0
        val_print = 0
        for gt_path, inp_path in data_list:
            _inp_data = imread(os.path.join('dataset', inp_path))  # (h, w, c)
            gt_data = imread(os.path.join('dataset', gt_path))  # (h, w, c)
            inp_data = _inp_data.astype('float32') / 255

            h = int(inp_data.shape[0])
            w = int(inp_data.shape[1])

            if (h % 16) == 0:
                new_h = h
            else:
                new_h = h - (h % 16) + 16
            if (w % 16) == 0:
                new_w = w
            else:
                new_w = w - (w % 16) + 16

            if network_dict['%dx%d' % (new_h,new_w)] == []:
                print('add network to dict', new_h, new_w)
                inputs = tf.placeholder(shape=[self.batch_size, new_h, new_w, inp_chns], dtype=tf.float32)
                outputs = self.generator(inputs, reuse=True)

                network_dict['%dx%d' % (new_h, new_w)] = [inputs, outputs]

            if (new_h - h) > 0 or (new_w - w) > 0:
                inp_data = np.pad(inp_data, ((0, new_h - h), (0, new_w - w), (0, 0)), 'edge')

            inp_data = np.expand_dims(inp_data, 0)
            if self.args.model == 'color':
                val_x_unwrap = sess.run(network_dict['%dx%d' % (new_h, new_w)][1], feed_dict={network_dict['%dx%d' % (new_h, new_w)][0]: inp_data})
                out = val_x_unwrap[-1]
            else:
                inp_data = np.transpose(inp_data, (3, 1, 2, 0))  # (c, h, w, 1)
                val_x_unwrap = sess.run(network_dict['%dx%d' % (new_h, new_w)][1], feed_dict={network_dict['%dx%d' % (new_h, new_w)][0]: inp_data})
                out = val_x_unwrap[-1]
                out = np.transpose(out, (3, 1, 2, 0))  # (1, h, w, c)

            if (new_h - h) > 0 or (new_w - w) > 0:
                out = out[:, :h, :w, :]


            #out = np.clip(out, 0, 1) * 255
            #out = out[0].astype('uint8')
            #gt_data = gt_data.astype('uint8')
            out = np.clip(out*255, 0, 255) + 0.5
            out = out.astype('uint8')
            out = out[0]

            display_img = np.hstack([_inp_data, out, gt_data])

            mse = np.mean((gt_data - out) ** 2)
            val_psnr = cv2.PSNR(gt_data, out)
            input_psnr = cv2.PSNR(gt_data, _inp_data)
            print(mse, input_psnr, val_psnr, val_print)
            val_print += 1
            total_psnr += val_psnr
            total_psnr_input += input_psnr
            total_mse += mse

            img_name = inp_path.replace('/','_')
            img_name = img_name.split('.')[0]
            dis_img_name = img_name + "_%.2f_%.2f" % (input_psnr, val_psnr) + ".jpg"
            #imsave(os.path.join(output_path, dis_img_name), display_img)

            out_img_name = inp_path.split('/')
            out_img_name = out_img_name[-1]
            if 'RealBlur_J' in self.args.datalist:
                out_img_name = inp_path.split('/')[1] + "_" + out_img_name
            elif 'BSD' in self.args.datalist:
                out_img_name = inp_path.split('/')[-6] + "_" + inp_path.split('/')[-4] + "_" + out_img_name
            elif 'gopro' in self.args.datalist:
                out_img_name = inp_path.split('/')[-3] + "_" + out_img_name
            elif 'kohler' in self.args.datalist:
                pass
            elif 'lai' in self.args.datalist:
                out_img_name = out_img_name[:-4] + '.png'
            else:
                out_img_name = inp_path.split('/')[-4] + "_" + inp_path.split('/')[-3] + "_" + out_img_name
            imsave(os.path.join(output_path, out_img_name), out)
            #imsave(os.path.join(output_path, out_img_name), _inp_data)

        mean_psnr = total_psnr / len(self.data_list)
        mean_psnr_input = total_psnr_input / len(self.data_list)
        total_loss = total_mse / len(self.data_list)

        format_str = ('%s: step %d, validation loss = (%.5f; %.5f, %.5f), psnr : (%.5f), psnr_input : (%.5f)')
        print(format_str % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), self.args.load_iteration, total_loss, 0.0, 0.0, mean_psnr, mean_psnr_input))
