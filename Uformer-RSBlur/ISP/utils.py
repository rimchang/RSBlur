import numpy as np
import os
import scipy.io
import math
import skimage
from ISP.Demosaicing_malvar2004_pytorch import Demosaic
import random
import glob
import torch


dir_name = os.path.dirname(os.path.abspath(__file__))

xyz2cam_realblur = scipy.io.loadmat(dir_name + '/mat_collections/a7r3_xyz2cam.mat')['final_xyz2cam'].astype('float32')
cam2xyz_realblur = scipy.io.loadmat(dir_name + '/mat_collections/a7r3_cam2xyz.mat')['final_cam2xyz'].astype('float32')

cam2xyz_realblur_right = cam2xyz_realblur.transpose(1, 0)
xyz2cam_realblur_right = xyz2cam_realblur.transpose(1, 0)

alpha_realblur = scipy.io.loadmat(dir_name + '/mat_collections/a7r3_polynomial_alpha.mat')['alpha'].astype('float32')
alpha_realblur_rgb2lin = scipy.io.loadmat(dir_name + '/mat_collections/a7r3_polynomial_alpha_srgb2lin.mat')['alpha'].astype('float32')

# lin2xyz
M_lin2xyz_np = scipy.io.loadmat(dir_name + '/mat_collections/M_lin2xyz.mat')['M'].astype('float32')

# xyz2lin
M_xyz2lin_np = scipy.io.loadmat(dir_name + '/mat_collections/M_xyz2lin.mat')['M'].astype('float32')

# lin2xyz
M_lin2xyz = torch.tensor(scipy.io.loadmat(dir_name + '/mat_collections/M_lin2xyz.mat')['M'], dtype=torch.float32)

# xyz2lin
M_xyz2lin = torch.tensor(scipy.io.loadmat(dir_name + '/mat_collections/M_xyz2lin.mat')['M'], dtype=torch.float32)

lookup_table = scipy.io.loadmat(dir_name + '/mat_collections/lookup_table.mat')['lookup_table'].astype('float32')

def rgb2lin(x):
    # based matlab rgb2lin
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


def lin2rgb(x):
    # based matlab lin2rgb
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

def rgb2lin_pt(x):
    gamma = 2.4
    a = 1 / 1.055
    b = 0.055 / 1.055
    c = 1 / 12.92
    d = 0.04045

    in_sign = -2 * torch.lt(x, 0) + 1
    abs_x = torch.abs(x)

    lin_range = torch.lt(abs_x ,d)
    gamma_range = torch.logical_not(lin_range)

    lin_value = (c * abs_x)
    gamma_value = torch.exp(gamma * torch.log(a * abs_x + b))

    new_x = (lin_value * lin_range) + (gamma_value * gamma_range)
    new_x = new_x * in_sign

    return new_x

def lin2rgb_pt(x):
    gamma = 1/2.4
    a = 1.055
    b = -0.055
    c = 12.92
    d = 0.0031308

    in_sign = -2 * torch.lt(x, 0) + 1
    abs_x = torch.abs(x)

    lin_range = torch.lt(abs_x ,d)
    gamma_range = torch.logical_not(lin_range)

    lin_range = lin_range
    gamma_range = gamma_range

    lin_value = (c * abs_x)
    gamma_value = a * torch.exp(gamma * torch.log(abs_x)) + b
    new_x = (lin_value * lin_range) + (gamma_value * gamma_range)
    new_x = new_x * in_sign

    return new_x


def lin2xyz(x):
    # based matlab rgb2xyz
    M = M_lin2xyz

    h, w, c = x.shape
    v = x.reshape(h*w, c)
    xyz = torch.matmul(v, M)
    xyz = xyz.reshape(h, w, c)

    return xyz

def xyz2lin(x):
    # based matlab rgb2xyz
    M = M_xyz2lin

    h, w, c = x.shape
    xyz = x.reshape(h*w, c)
    lin_rgb = torch.matmul(xyz, M)
    lin_rgb = lin_rgb.reshape(h, w, c)

    return lin_rgb

def apply_cmatrix(img, matrix):
    # img : (h, w, c)
    # matrix : (3, 3)

    """
    same results below code
    img_reshape = img.reshape(1, h*w, 3)
    out2 = torch.matmul(img_reshape, matrix.permute(0, 2, 1))
    out2 = out2.reshape(1, h, w, 3)
    """

    images = img[:,:,None,:] # (h, w, 1, c)
    ccms = matrix[None, None, :, :] # (1, 1, 3, 3)
    out = torch.sum(images * ccms, -1) # (h, w, 3)

    return out

def mosaic_bayer(image, pattern):

    """Extracts RGGB Bayer planes from an RGB image."""
    shape = image.shape # (h, w, c)

    if pattern == 'RGGB':
        red = image[0::2, 0::2, 0] # (h/2, w/2)
        green_red = image[0::2, 1::2, 1]
        green_blue = image[1::2, 0::2, 1]
        blue = image[1::2, 1::2, 2]
    elif pattern == 'BGGR':
        red = image[0::2, 0::2, 2] # (h/2, w/2)
        green_red = image[0::2, 1::2, 1]
        green_blue = image[1::2, 0::2, 1]
        blue = image[1::2, 1::2, 0]
    elif pattern == 'GRBG':
        red = image[0::2, 0::2, 1] # (h/2, w/2)
        green_red = image[0::2, 1::2, 0]
        green_blue = image[1::2, 0::2, 2]
        blue = image[1::2, 1::2, 1]
    elif pattern == 'GBRG':
        red = image[0::2, 0::2, 1] # (h/2, w/2)
        green_red = image[0::2, 1::2, 2]
        green_blue = image[1::2, 0::2, 0]
        blue = image[1::2, 1::2, 1]

    image = torch.stack((red, green_red, green_blue, blue), dim=2)  # (h/2, w/2, 4)
    image = image.view(shape[0] // 2, shape[1] // 2, 4)

    return image

def WB_img(img, pattern, fr_now, fb_now):

    red_gains = fr_now
    blue_gains = fb_now
    green_gains = 1.0

    if pattern == 'RGGB':
        gains = torch.tensor([red_gains, green_gains, green_gains, blue_gains]).float()
    elif pattern == 'BGGR':
        gains = torch.tensor([blue_gains, green_gains, green_gains, red_gains]).float()
    elif pattern == 'GRBG':
        gains = torch.tensor([green_gains, red_gains, blue_gains, green_gains]).float()
    elif pattern == 'GBRG':
        gains = torch.tensor([green_gains, blue_gains, red_gains, green_gains]).float()

    gains = gains[None, None, :]
    img = img * gains

    return img


def rgb2lin_a7r3(img):
    img = img.numpy()

    x = img * 255.0
    v = np.interp(x, lookup_table[:, 1], lookup_table[:, 0]) / 255.0
    v = np.clip(v, 0, 1).astype('float32')
    return torch.from_numpy(v)

def lin2rgb_a7r3(img):
    img = img.numpy()

    x = img * 255.0
    v = np.interp(x, lookup_table[:, 0], lookup_table[:, 1]) / 255.0
    v = np.clip(v, 0, 1).astype('float32')
    return torch.from_numpy(v)

def add_Poisson_noise_random(img, beta1, beta2):

    h, w, c = img.shape

    # bsd : 2.3282e-05, my : 0.0001
    min_beta1 = beta1 * 0.5
    random_K_v = min_beta1 + torch.rand(1) * (beta1 * 1.5 - min_beta1)
    random_K_v = random_K_v.view(1, 1, 1).to(img.device)

    noisy_img = torch.poisson(img / random_K_v)
    noisy_img = noisy_img * random_K_v

    # bsd : 1.9452e-04, my : 9.1504e-04
    min_beta2 = beta2 * 0.5
    random_other = min_beta2 + torch.rand(1) * (beta2 * 1.5 - min_beta2)
    random_other = random_other.view(1, 1, 1).to(img.device)

    noisy_img = noisy_img + (torch.normal(torch.zeros_like(noisy_img), std=1)*random_other)

    return noisy_img

def lin2rgb_a7r3_polynomial(img):
    srgb = alpha_realblur[0][0] + \
           alpha_realblur[1][0] * img + \
           alpha_realblur[2][0] * torch.pow(img, 2) + \
           alpha_realblur[3][0] * torch.pow(img, 3) + \
           alpha_realblur[4][0] * torch.pow(img, 4) + \
           alpha_realblur[5][0] * torch.pow(img, 5) + \
           alpha_realblur[6][0] * torch.pow(img, 6) + \
           alpha_realblur[7][0] * torch.pow(img, 7)

    srgb = torch.clamp(srgb, 0, 1)

    return srgb