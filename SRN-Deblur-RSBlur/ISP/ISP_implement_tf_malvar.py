import numpy as np
import cv2
import os
import h5py
import scipy.io
import math
import skimage
from ISP.Demosaicing_malvar2004_tf import demosaicing_CFA_Bayer_Malvar2004_tf
import random
import glob
import tensorflow as tf

# the codes are based on ISP process of CBDNet (https://github.com/GuoShi28/CBDNet/tree/master/SomeISP_operator_python)

def rgb2lin_np(x):
    # based on matlab rgb2lin, we use similar function as gamma correction of 2.2.
    # refer to Computer displays section on (https://en.wikipedia.org/wiki/Gamma_correction)

    gamma = 2.4
    a = 1 / 1.055
    b = 0.055 / 1.055
    c = 1 / 12.92
    d = 0.04045

    in_sign = -2 * (x < 0) + 1
    abs_x = np.abs(x)

    lin_range = (abs_x < d)
    gamma_range = np.logical_not(lin_range)

    out = x.copy()
    out[lin_range] = c * abs_x[lin_range]
    out[gamma_range] = np.exp(gamma * np.log(a * abs_x[gamma_range] + b))

    out = out * in_sign

    return out


def lin2rgb_np(x):
    # based on matlab lin2rgb, we use similar function as gamma correction of 2.2.
    # refer to Computer displays section on (https://en.wikipedia.org/wiki/Gamma_correction)

    gamma = 1 / 2.4
    a = 1.055
    b = -0.055
    c = 12.92
    d = 0.0031308

    in_sign = -2 * (x < 0) + 1
    abs_x = np.abs(x)

    lin_range = (abs_x < d)
    gamma_range = np.logical_not(lin_range)

    out = x.copy()
    out[lin_range] = c * abs_x[lin_range]
    out[gamma_range] = a * np.exp(gamma * np.log(abs_x[gamma_range])) + b

    out = out * in_sign

    return out

def rgb2lin_tf(x):
    # based on matlab rgb2lin, we use similar function as gamma correction of 2.2.
    # refer to Computer displays section on (https://en.wikipedia.org/wiki/Gamma_correction)

    gamma = 2.4
    a = 1 / 1.055
    b = 0.055 / 1.055
    c = 1 / 12.92
    d = 0.04045

    in_sign = -2 * tf.cast(tf.math.less(x, 0), tf.float32) + 1
    abs_x = tf.abs(x)

    lin_range = tf.math.less(abs_x ,d)
    gamma_range = tf.math.logical_not(lin_range)

    lin_range = tf.cast(lin_range, tf.float32)
    gamma_range = tf.cast(gamma_range, tf.float32)

    lin_value = (c * abs_x)
    gamma_value = tf.exp(gamma * tf.log(a * abs_x + b))
    new_x = (lin_value * lin_range) + (gamma_value * gamma_range)
    new_x = new_x * in_sign

    return new_x

def lin2rgb_tf(x):
    # based on matlab lin2rgb, we use similar function as gamma correction of 2.2.
    # refer to Computer displays section on (https://en.wikipedia.org/wiki/Gamma_correction)

    gamma = 1/2.4
    a = 1.055
    b = -0.055
    c = 12.92
    d = 0.0031308

    in_sign = -2 * tf.cast(tf.math.less(x, 0), tf.float32) + 1
    abs_x = tf.abs(x)

    lin_range = tf.math.less(abs_x ,d)
    gamma_range = tf.math.logical_not(lin_range)

    lin_range = tf.cast(lin_range, tf.float32)
    gamma_range = tf.cast(gamma_range, tf.float32)

    lin_value = (c * abs_x)
    gamma_value = a * tf.exp(gamma * tf.log(abs_x)) + b
    new_x = (lin_value * lin_range) + (gamma_value * gamma_range)
    new_x = new_x * in_sign

    return new_x


class ISP:
    def __init__(self, random_crf=False, random_wb=False):
        # ccm matrix for the RealBlur dataset
        self.xyz2cam_realblur = scipy.io.loadmat('ISP/mat_collections/a7r3_xyz2cam.mat')['final_xyz2cam']
        self.cam2xyz_realblur = scipy.io.loadmat('ISP/mat_collections/a7r3_cam2xyz.mat')['final_cam2xyz']

        # estimated crf for the RealBlur dataset
        self.alpha_realblur = scipy.io.loadmat('ISP/mat_collections/a7r3_polynomial_alpha.mat')['alpha']

        # lin2xyz matrix extracted from matlab
        self.M_lin2xyz_np = scipy.io.loadmat('ISP/mat_collections/M_lin2xyz.mat')['M']

        # xyz2lin matrix extracted from matlab
        self.M_xyz2lin_np = scipy.io.loadmat('ISP/mat_collections/M_xyz2lin.mat')['M']

        # lin2xyz for tf
        self.M_lin2xyz = tf.constant(scipy.io.loadmat('ISP/mat_collections/M_lin2xyz.mat')['M'], dtype='float32')

        # xyz2lin for tf
        self.M_xyz2lin = tf.constant(scipy.io.loadmat('ISP/mat_collections/M_xyz2lin.mat')['M'], dtype='float32')

        # adaptXYZ_d502d65 matrix extracted from matlab
        self.M_d502d65 = tf.constant(scipy.io.loadmat('ISP/mat_collections/M_d502d65.mat')['M'], dtype='float32')

        # adaptXYZ_d652d50 matrix extracted from matlab
        self.M_d652d50 = tf.constant(scipy.io.loadmat('ISP/mat_collections/M_d652d50.mat')['M'], dtype='float32')

        # ccm collections for the RSBlur dataset
        cam2xyz_path = glob.glob('ISP/ccm_collections/**/ccm.mat')
        self.cam2xyz_list = {}
        self.xyz2cam_list = {}
        for cam2xyz_path in cam2xyz_path:
            key = cam2xyz_path.split('/')[-2]
            cam2xyz_matrix = scipy.io.loadmat(cam2xyz_path)
            self.cam2xyz_list[key] = cam2xyz_matrix['colorCorrectionMatrix'][0, 1]
            self.xyz2cam_list[key] = np.linalg.inv(cam2xyz_matrix['colorCorrectionMatrix'][0, 1])
        self.cam2xyz_keys = self.cam2xyz_list.keys()

        # wb collections for the RSBlur dataset
        wbs_path = glob.glob('ISP/ccm_collections/**/wb_params.mat')
        self.wbs = {}
        for wb_path in wbs_path:
            key = wb_path.split('/')[-2]
            wb_value = scipy.io.loadmat(wb_path)
            self.wbs[key] = wb_value['wb_param'][0, 3:]
        self.wbs_keys = self.wbs.keys()

        # noise params
        # RSBlur : 0.0001, realblur : 8.8915e-05
        # RSBlur : 9.1504e-04, realblur : 2.9430e-05

        self.beta1 = 0.0001
        self.beta2 = 9.1504e-04


    def lin2xyz_np(self, x):
        # based matlab rgb2xyz
        M = self.M_lin2xyz_np
        xyz = np.matmul(x, M)
        return xyz

    def xyz2lin_np(self, x):
        # based matlab rgb2xyz
        M = self.M_xyz2lin_np
        lin_rgb = np.matmul(x, M)

        return lin_rgb

    def adaptXYZ_d502d65(self, xyz_in):
        xyz_out = tf.matmul(xyz_in, tf.transpose(self.M_d502d65))
        return xyz_out

    def adaptXYZ_d652d50(self, xyz_in):
        xyz_out = tf.matmul(xyz_in, tf.transpose(self.M_d652d50))
        return xyz_out

    def lin2xyz(self, x, d652d50=True):
        # based matlab rgb2xyz
        M = self.M_lin2xyz

        b, h, w, c = x.get_shape().as_list()
        v = tf.reshape(x, [-1, h * w, c])
        xyz = tf.matmul(v, M)
        if d652d50:
            xyz = self.adaptXYZ_d652d50(xyz)
        xyz = tf.reshape(xyz, [-1, h, w, c])
        return xyz

    def xyz2lin(self, x, d502d65=True):
        # based matlab rgb2xyz
        M = self.M_xyz2lin

        b, h, w, c = x.get_shape().as_list()
        xyz = tf.reshape(x, [-1, h * w, c])
        if d502d65:
            xyz = self.adaptXYZ_d502d65(xyz)
        lin_rgb = tf.matmul(xyz, M)
        lin_rgb = tf.reshape(lin_rgb, [-1, h, w, c])

        return lin_rgb


    def lin2rgb_realblur(self, img):
        srgb = self.alpha_realblur[0] + \
               self.alpha_realblur[1] * img + \
               self.alpha_realblur[2] * tf.pow(img, 2) + \
               self.alpha_realblur[3] * tf.pow(img, 3) + \
               self.alpha_realblur[4] * tf.pow(img, 4) + \
               self.alpha_realblur[5] * tf.pow(img, 5) + \
               self.alpha_realblur[6] * tf.pow(img, 6) + \
               self.alpha_realblur[7] * tf.pow(img, 7)

        srgb = tf.clip_by_value(srgb, 0, 1)

        return srgb

    def apply_cmatrix(self, img, matrix):
        images = img[:, :, :, tf.newaxis, :]
        ccms = matrix[:, tf.newaxis, tf.newaxis, :, :]
        return tf.reduce_sum(images * ccms, axis=-1)

    def mosaic_bayer(self, image, pattern):

        """Extracts RGGB Bayer planes from an RGB image."""
        image.shape.assert_is_compatible_with((None, None, None, 3))
        shape = tf.shape(image)

        if pattern == 'RGGB':
            red = image[:, 0::2, 0::2, 0] # (b, h/2, w/2)
            green_red = image[:, 0::2, 1::2, 1]
            green_blue = image[:, 1::2, 0::2, 1]
            blue = image[:, 1::2, 1::2, 2]
        elif pattern == 'BGGR':
            red = image[:, 0::2, 0::2, 2] # (b, h/2, w/2)
            green_red = image[:, 0::2, 1::2, 1]
            green_blue = image[:, 1::2, 0::2, 1]
            blue = image[:, 1::2, 1::2, 0]
        elif pattern == 'GRBG':
            red = image[:, 0::2, 0::2, 1] # (b, h/2, w/2)
            green_red = image[:, 0::2, 1::2, 0]
            green_blue = image[:, 1::2, 0::2, 2]
            blue = image[:, 1::2, 1::2, 1]
        elif pattern == 'GBRG':
            red = image[:, 0::2, 0::2, 1] # (b, h/2, w/2)
            green_red = image[:, 0::2, 1::2, 2]
            green_blue = image[:, 1::2, 0::2, 0]
            blue = image[:, 1::2, 1::2, 1]

        image = tf.stack((red, green_red, green_blue, blue), axis=-1)  # (b, h/2, w/2, 4)
        image = tf.reshape(image, (-1, shape[1] // 2, shape[2] // 2, 4))

        return image

    def WB_img(self, img, pattern, fr_now, fb_now):

        red_gains = fr_now
        blue_gains = fb_now
        green_gains = tf.ones_like(red_gains)

        if pattern == 'RGGB':
            gains = tf.stack([red_gains, green_gains, green_gains, blue_gains], axis=-1)
        elif pattern == 'BGGR':
            gains = tf.stack([blue_gains, green_gains, green_gains, red_gains], axis=-1)
        elif pattern == 'GRBG':
            gains = tf.stack([green_gains, red_gains, blue_gains, green_gains], axis=-1)
        elif pattern == 'GBRG':
            gains = tf.stack([green_gains, blue_gains, red_gains, green_gains], axis=-1)

        gains = gains[:, tf.newaxis, tf.newaxis, :]
        img = img * gains

        return img

    def add_Poisson_noise_random(self, img):

        b, h, w, c = img.get_shape().as_list()

        # signal-dependent noise
        min_beta1 = self.beta1 * 0.5
        random_K_v = min_beta1 + tf.random.uniform([b], 0, 1) * (self.beta1 * 1.5 - min_beta1)
        random_K_v = tf.reshape(random_K_v, [b, 1, 1, 1])

        noisy_img = tf.random.poisson(img / random_K_v, [1])
        noisy_img = noisy_img[0,:,:,:,:] * random_K_v

        # signal-independent noise
        min_beta2 = self.beta2 * 0.5
        random_other = min_beta2 + tf.random.uniform([b], 0, 1) * (self.beta2 * 1.5 - min_beta2)
        random_other = tf.reshape(random_other, [b, 1, 1, 1])

        noisy_img = noisy_img + (tf.random_normal(shape=tf.shape(noisy_img), mean=0.0, stddev=1, dtype=tf.float32)*random_other)

        return noisy_img

    def poisson_gamma(self, img, M_xyz2cam, fr_now, fb_now):
        img_rgb = img
        batch_size, _, _, _ = img_rgb.get_shape().as_list()

        # -------- INVERSE ISP PROCESS -------------------
        # Step 1 : inverse tone mapping
        img_L = rgb2lin_tf(img_rgb)

        # Step 2 : from RGB to XYZ
        img_XYZ = self.lin2xyz(img_L, False)

        # Step 3: from XYZ to Cam
        img_Cam = self.apply_cmatrix(img_XYZ, M_xyz2cam)
        img_Cam = tf.clip_by_value(img_Cam, 0, 1)

        # Step 4: Mosaic and inverse white balance
        img_Cam_split = tf.split(img_Cam, 4, axis=0)
        fr_now_split = tf.split(fr_now, 4, axis=0)
        fb_now_split = tf.split(fb_now, 4, axis=0)
        img_mosaic_rggb = self.mosaic_bayer(img_Cam_split[0], 'RGGB')
        img_mosaic_rggb = self.WB_img(img_mosaic_rggb, 'RGGB', fr_now_split[0], fb_now_split[0])

        img_mosaic_bggr = self.mosaic_bayer(img_Cam_split[1], 'BGGR')
        img_mosaic_bggr = self.WB_img(img_mosaic_bggr, 'BGGR', fr_now_split[1], fb_now_split[1])

        img_mosaic_grbg = self.mosaic_bayer(img_Cam_split[2], 'GRBG')
        img_mosaic_grbg = self.WB_img(img_mosaic_grbg, 'GRBG', fr_now_split[2], fb_now_split[2])

        img_mosaic_gbrg = self.mosaic_bayer(img_Cam_split[3], 'GBRG')
        img_mosaic_gbrg = self.WB_img(img_mosaic_gbrg, 'GBRG', fr_now_split[3], fb_now_split[3])
        img_mosaic = tf.concat([img_mosaic_rggb, img_mosaic_bggr, img_mosaic_grbg, img_mosaic_gbrg], axis=0) # (b, h/2, w/2, 4)

        # -------- ADDING POISSON-GAUSSIAN NOISE ON RAW -
        img_mosaic = self.add_Poisson_noise_random(img_mosaic)

        # -------- ISP PROCESS --------------------------
        # Step 4 : White balance and Demosiac
        img_mosaic_split = tf.split(img_mosaic, 4, axis=0)
        img_demosaic_rggb = self.WB_img(img_mosaic_split[0], 'RGGB', 1/fr_now_split[0], 1/fb_now_split[0])
        img_demosaic_rggb = tf.depth_to_space(img_demosaic_rggb, 2)
        img_demosaic_rggb = demosaicing_CFA_Bayer_Malvar2004_tf(img_demosaic_rggb, pattern='RGGB', batch_size=batch_size//4)

        img_demosaic_bggr = self.WB_img(img_mosaic_split[1], 'BGGR', 1/fr_now_split[1], 1/fb_now_split[1])
        img_demosaic_bggr = tf.depth_to_space(img_demosaic_bggr, 2)
        img_demosaic_bggr = demosaicing_CFA_Bayer_Malvar2004_tf(img_demosaic_bggr, pattern='BGGR', batch_size=batch_size//4)

        img_demosaic_grbg = self.WB_img(img_mosaic_split[2], 'GRBG', 1/fr_now_split[2], 1/fb_now_split[2])
        img_demosaic_grbg = tf.depth_to_space(img_demosaic_grbg, 2)
        img_demosaic_grbg = demosaicing_CFA_Bayer_Malvar2004_tf(img_demosaic_grbg, pattern='GRBG', batch_size=batch_size//4)

        img_demosaic_gbrg = self.WB_img(img_mosaic_split[3], 'GBRG', 1/fr_now_split[3], 1/fb_now_split[3])
        img_demosaic_gbrg = tf.depth_to_space(img_demosaic_gbrg, 2)
        img_demosaic_gbrg = demosaicing_CFA_Bayer_Malvar2004_tf(img_demosaic_gbrg, pattern='GBRG', batch_size=batch_size//4)

        img_demosaic = tf.concat([img_demosaic_rggb, img_demosaic_bggr, img_demosaic_grbg, img_demosaic_gbrg], axis=0)
        img_demosaic = tf.clip_by_value(img_demosaic, 0, 1)

        # Step 3 : from Cam to XYZ
        img_IXYZ = self.apply_cmatrix(img_demosaic, tf.linalg.inv(M_xyz2cam))

        # Step 2 : frome XYZ to RGB
        img_IL = self.xyz2lin(img_IXYZ, False)

        # Step 1 : tone mapping
        img_Irgb = lin2rgb_tf(img_IL)
        img_Irgb = tf.clip_by_value(img_Irgb, 0, 1)

        return img_Irgb

    def poisson_RSBlur(self, img, M_xyz2cam, fr_now, fb_now):
        img_rgb = img
        batch_size, _, _, _ = img_rgb.get_shape().as_list()

        # -------- INVERSE ISP PROCESS -------------------
        # Step 1 : inverse tone mapping
        img_L = rgb2lin_tf(img_rgb)

        # Step 2 : from RGB to XYZ
        img_XYZ = self.lin2xyz(img_L, True)

        # Step 3: from XYZ to Cam
        img_Cam = self.apply_cmatrix(img_XYZ, M_xyz2cam) # CCM of RSBlur estimated on XYZ space
        img_Cam = self.xyz2lin(img_Cam, True)
        img_Cam = tf.clip_by_value(img_Cam, 0, 1)

        # Step 4: Mosaic and inverse white balance
        img_Cam_split = tf.split(img_Cam, 4, axis=0)
        fr_now_split = tf.split(fr_now, 4, axis=0)
        fb_now_split = tf.split(fb_now, 4, axis=0)
        img_mosaic_rggb = self.mosaic_bayer(img_Cam_split[0], 'RGGB')
        img_mosaic_rggb = self.WB_img(img_mosaic_rggb, 'RGGB', fr_now_split[0], fb_now_split[0])

        img_mosaic_bggr = self.mosaic_bayer(img_Cam_split[1], 'BGGR')
        img_mosaic_bggr = self.WB_img(img_mosaic_bggr, 'BGGR', fr_now_split[1], fb_now_split[1])

        img_mosaic_grbg = self.mosaic_bayer(img_Cam_split[2], 'GRBG')
        img_mosaic_grbg = self.WB_img(img_mosaic_grbg, 'GRBG', fr_now_split[2], fb_now_split[2])

        img_mosaic_gbrg = self.mosaic_bayer(img_Cam_split[3], 'GBRG')
        img_mosaic_gbrg = self.WB_img(img_mosaic_gbrg, 'GBRG', fr_now_split[3], fb_now_split[3])
        img_mosaic = tf.concat([img_mosaic_rggb, img_mosaic_bggr, img_mosaic_grbg, img_mosaic_gbrg], axis=0) # (b, h/2, w/2, 4)

        # -------- ADDING POISSON-GAUSSIAN NOISE ON RAW -
        img_mosaic = self.add_Poisson_noise_random(img_mosaic)

        # -------- ISP PROCESS --------------------------
        # Step 4 : White balance and Demosiac
        img_mosaic_split = tf.split(img_mosaic, 4, axis=0)
        img_demosaic_rggb = self.WB_img(img_mosaic_split[0], 'RGGB', 1/fr_now_split[0], 1/fb_now_split[0])
        img_demosaic_rggb = tf.depth_to_space(img_demosaic_rggb, 2)
        img_demosaic_rggb = demosaicing_CFA_Bayer_Malvar2004_tf(img_demosaic_rggb, pattern='RGGB', batch_size=batch_size//4)

        img_demosaic_bggr = self.WB_img(img_mosaic_split[1], 'BGGR', 1/fr_now_split[1], 1/fb_now_split[1])
        img_demosaic_bggr = tf.depth_to_space(img_demosaic_bggr, 2)
        img_demosaic_bggr = demosaicing_CFA_Bayer_Malvar2004_tf(img_demosaic_bggr, pattern='BGGR', batch_size=batch_size//4)

        img_demosaic_grbg = self.WB_img(img_mosaic_split[2], 'GRBG', 1/fr_now_split[2], 1/fb_now_split[2])
        img_demosaic_grbg = tf.depth_to_space(img_demosaic_grbg, 2)
        img_demosaic_grbg = demosaicing_CFA_Bayer_Malvar2004_tf(img_demosaic_grbg, pattern='GRBG', batch_size=batch_size//4)

        img_demosaic_gbrg = self.WB_img(img_mosaic_split[3], 'GBRG', 1/fr_now_split[3], 1/fb_now_split[3])
        img_demosaic_gbrg = tf.depth_to_space(img_demosaic_gbrg, 2)
        img_demosaic_gbrg = demosaicing_CFA_Bayer_Malvar2004_tf(img_demosaic_gbrg, pattern='GBRG', batch_size=batch_size//4)

        img_demosaic = tf.concat([img_demosaic_rggb, img_demosaic_bggr, img_demosaic_grbg, img_demosaic_gbrg], axis=0)
        img_demosaic = tf.clip_by_value(img_demosaic, 0, 1)

        # Step 3 : from Cam to XYZ
        img_IXYZ = self.lin2xyz(img_demosaic, True)  # d652d50
        img_IXYZ = self.apply_cmatrix(img_IXYZ, tf.linalg.inv(M_xyz2cam))

        # Step 2 : frome XYZ to RGB
        img_IL = self.xyz2lin(img_IXYZ, True) #d502d65

        # Step 1 : tone mapping
        img_Irgb = lin2rgb_tf(img_IL)
        img_Irgb = tf.clip_by_value(img_Irgb, 0, 1)

        return img_Irgb