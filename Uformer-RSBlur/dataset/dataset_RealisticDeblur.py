import os
import torch
import numpy as np
from PIL import Image as Image
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import cv2
import random
from ISP.utils import *


def get_realisticGoProUtraining_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return RealisticGoProUDataset(rgb_dir, patch_size=img_options['patch_size'])

def get_naiveGoProUtraining_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return RealisticGoProUDataset(rgb_dir, patch_size=img_options['patch_size'], realistic_pipeline=False)

def get_realisticGoProABMEtraining_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return RealisticGoProABMEDataset(rgb_dir, patch_size=img_options['patch_size'])

def get_naiveGoProABMEtraining_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return RealisticGoProABMEDataset(rgb_dir, patch_size=img_options['patch_size'], realistic_pipeline=False)

def get_validation_deblur_data(rgb_dir, img_options=None):
    assert os.path.exists(rgb_dir)
    image_dir = os.path.join(rgb_dir, 'test')

    return RealBlurDataset(image_dir, center_crop=True)

def get_test_data(rgb_dir, img_options=None):
    assert os.path.exists(rgb_dir)
    image_dir = os.path.join(rgb_dir, 'test')

    return RealBlurDataset(image_dir)

class RealisticGoProABMEDataset(Dataset):
    def __init__(self, image_dir, patch_size=256, image_aug=True, realistic_pipeline=True):
        self.image_dir = image_dir
        self.image_list = glob.glob(os.path.join(image_dir, '**/**/avg_inter_img/*.png'))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.ps = patch_size
        self.realistic_pipeline = realistic_pipeline

        self.image_aug = image_aug
        self.demosaic = Demosaic()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        # read sharp image
        label = cv2.imread(self.image_list[idx].replace('/avg_inter_img/avg_blur.png', '/gt/gt_sharp.png'),
                           cv2.IMREAD_COLOR).astype('float32') / 255
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        # read blurred image
        blurred = cv2.imread(self.image_list[idx], cv2.IMREAD_COLOR).astype('float32') / 255
        blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)

        # read saturation mask
        sat_mask = cv2.imread(self.image_list[idx].replace('/avg_inter_img/', '/avg_inter_mask_100/'),
                              cv2.IMREAD_COLOR).astype('float32') / 255
        sat_mask = cv2.cvtColor(sat_mask, cv2.COLOR_BGR2RGB)

        # numpy to torch
        label_pt = torch.from_numpy(label).float()
        blurred_pt = torch.from_numpy(blurred).float()
        sat_mask_pt = torch.from_numpy(sat_mask).float()

        # random crop
        # Due to artifacts of demosaic on edges, we crop bigger images.
        boundary_size = 8
        ps = self.ps + boundary_size
        hh, ww = label_pt.shape[0], label_pt.shape[1]

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)

        label_pt = label_pt[rr:rr + ps, cc:cc + ps, :]
        blurred_pt = blurred_pt[rr:rr + ps, cc:cc + ps, :]
        sat_mask_pt = sat_mask_pt[rr:rr + ps, cc:cc + ps, :]

        # random image augmentation
        if self.image_aug:
            aug = random.randint(0, 8)
            if aug == 1:
                label_pt = label_pt.flip(0)
            elif aug == 2:
                label_pt = label_pt.flip(1)
            elif aug == 3:
                label_pt = torch.rot90(label_pt, dims=(0, 1))
            elif aug == 4:
                label_pt = torch.rot90(label_pt, dims=(0, 1), k=2)
            elif aug == 5:
                label_pt = torch.rot90(label_pt, dims=(0, 1), k=3)
            elif aug == 6:
                label_pt = torch.rot90(label_pt.flip(0), dims=(0, 1))
            elif aug == 7:
                label_pt = torch.rot90(label_pt.flip(1), dims=(0, 1))

            if aug == 1:
                blurred_pt = blurred_pt.flip(0)
            elif aug == 2:
                blurred_pt = blurred_pt.flip(1)
            elif aug == 3:
                blurred_pt = torch.rot90(blurred_pt, dims=(0, 1))
            elif aug == 4:
                blurred_pt = torch.rot90(blurred_pt, dims=(0, 1), k=2)
            elif aug == 5:
                blurred_pt = torch.rot90(blurred_pt, dims=(0, 1), k=3)
            elif aug == 6:
                blurred_pt = torch.rot90(blurred_pt.flip(0), dims=(0, 1))
            elif aug == 7:
                blurred_pt = torch.rot90(blurred_pt.flip(1), dims=(0, 1))

            if aug == 1:
                sat_mask_pt = sat_mask_pt.flip(0)
            elif aug == 2:
                sat_mask_pt = sat_mask_pt.flip(1)
            elif aug == 3:
                sat_mask_pt = torch.rot90(sat_mask_pt, dims=(0, 1))
            elif aug == 4:
                sat_mask_pt = torch.rot90(sat_mask_pt, dims=(0, 1), k=2)
            elif aug == 5:
                sat_mask_pt = torch.rot90(sat_mask_pt, dims=(0, 1), k=3)
            elif aug == 6:
                sat_mask_pt = torch.rot90(sat_mask_pt.flip(0), dims=(0, 1))
            elif aug == 7:
                sat_mask_pt = torch.rot90(sat_mask_pt.flip(1), dims=(0, 1))

        if not self.realistic_pipeline:

            # Naive synthesis
            blurred = blurred_pt
            gt = label_pt

            # crop boundary of blurred images
            blurred = blurred[boundary_size // 2:-boundary_size // 2, boundary_size // 2:-boundary_size // 2, :]
            gt = gt[boundary_size // 2:-boundary_size // 2, boundary_size // 2:-boundary_size // 2, :]

            # to_tensor
            blurred = blurred.permute((2, 0, 1)).contiguous()
            gt = gt.permute((2, 0, 1)).contiguous()

            return gt, blurred

        # -------- RSBlur Pipeline -----------------------
        # -------- INVERSE ISP PROCESS -------------------
        # inverse tone mapping
        blurred_L = rgb2lin_pt(blurred_pt)

        # saturation synthesis
        alpha_saturation = random.uniform(0.25, 1.75)
        blurred_L = blurred_L + (alpha_saturation * sat_mask_pt)
        blurred_L = torch.clamp(blurred_L, 0, 1)

        blurred_sat = blurred_L.clone()

        # from linear RGB to XYZ
        img_XYZ = lin2xyz(blurred_L)  # XYZ

        # from XYZ to Cam
        img_Cam = apply_cmatrix(img_XYZ, xyz2cam_realblur)  # raw RGB

        # Mosaic
        bayer_pattern = random.choice(['RGGB', 'BGGR', 'GRBG', 'GBRG'])
        img_mosaic = mosaic_bayer(img_Cam, bayer_pattern)

        # inverse white balance
        red_gain = random.uniform(1.9, 2.4)
        blue_gain = random.uniform(1.5, 1.9)
        img_mosaic = WB_img(img_mosaic, bayer_pattern, 1 / red_gain, 1 / blue_gain)

        # -------- ADDING POISSON-GAUSSIAN NOISE ON RAW -
        # estimated noise for the RealBlur dataset
        beta1 = 8.8915e-05
        beta2 = 2.9430e-05

        img_mosaic_noise = add_Poisson_noise_random(img_mosaic, beta1, beta2)

        # -------- ISP PROCESS --------------------------
        # White balance
        img_demosaic = WB_img(img_mosaic_noise, bayer_pattern, red_gain, blue_gain)

        # demosaic
        img_demosaic = torch.nn.functional.pixel_shuffle(img_demosaic.permute(2, 0, 1).unsqueeze(0), 2)
        img_demosaic = self.demosaic.forward(img_demosaic, pattern=bayer_pattern).squeeze(0).permute(1, 2, 0)

        # from Cam to XYZ
        img_IXYZ = apply_cmatrix(img_demosaic, cam2xyz_realblur)

        # frome XYZ to linear RGB
        img_IL = xyz2lin(img_IXYZ)

        # tone mapping
        img_Irgb = lin2rgb_pt(img_IL)
        # img_Irgb = lin2rgb_a7r3_polynomial(img_IL)
        img_Irgb = torch.clamp(img_Irgb, 0, 1)  # (h, w, c)

        blurred = img_Irgb
        gt = label_pt

        # don't add noise on saturated region
        sat_region = torch.ge(blurred_sat, 1.0)
        non_sat_region = torch.logical_not(sat_region)
        blurred = (blurred_sat * sat_region) + (blurred * non_sat_region)

        # Adopt a7r3 CRF
        gt = lin2rgb_a7r3_polynomial(rgb2lin_pt(gt))
        blurred = lin2rgb_a7r3_polynomial(rgb2lin_pt(blurred))

        # crop boundary of blurred images
        blurred = blurred[boundary_size // 2:-boundary_size // 2, boundary_size // 2:-boundary_size // 2, :]
        gt = gt[boundary_size // 2:-boundary_size // 2, boundary_size // 2:-boundary_size // 2, :]

        # to_tensor
        blurred = blurred.permute((2, 0, 1)).contiguous()
        gt = gt.permute((2, 0, 1)).contiguous()

        return gt, blurred

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError


class RealisticGoProUDataset(Dataset):
    def __init__(self, image_dir, patch_size=256, image_aug=True, realistic_pipeline=True):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'centroid_blurred_img/'))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.ps = patch_size
        self.realistic_pipeline = realistic_pipeline

        self.image_aug = image_aug
        self.demosaic = Demosaic()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        # read a sharp image
        label = cv2.imread(
            os.path.join(self.image_dir, 'target_img', self.image_list[idx].replace('_blurred.png', '_gt.png')),
            cv2.IMREAD_COLOR).astype(
            'float32') / 255
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        # read a blurred image
        blurred = cv2.imread(os.path.join(self.image_dir, 'centroid_blurred_img', self.image_list[idx]),
                             cv2.IMREAD_COLOR).astype(
            'float32') / 255
        blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)

        # read a saturation mask
        sat_mask = cv2.imread(os.path.join(self.image_dir, 'centroid_blurred_mask_100', self.image_list[idx]),
                              cv2.IMREAD_COLOR).astype(
            'float32') / 255
        sat_mask = cv2.cvtColor(sat_mask, cv2.COLOR_BGR2RGB)

        # numpy to torch
        label_pt = torch.from_numpy(label).float()
        blurred_pt = torch.from_numpy(blurred).float()
        sat_mask_pt = torch.from_numpy(sat_mask).float()

        # random crop
        # Due to artifacts of demosaic on edges, we crop bigger images.
        boundary_size = 8
        ps = self.ps + boundary_size
        hh, ww = label_pt.shape[0], label_pt.shape[1]

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)

        label_pt = label_pt[rr:rr + ps, cc:cc + ps, :]
        blurred_pt = blurred_pt[rr:rr + ps, cc:cc + ps, :]
        sat_mask_pt = sat_mask_pt[rr:rr + ps, cc:cc + ps, :]

        # random image augmentation
        if self.image_aug:
            aug = random.randint(0, 8)
            if aug == 1:
                label_pt = label_pt.flip(0)
            elif aug == 2:
                label_pt = label_pt.flip(1)
            elif aug == 3:
                label_pt = torch.rot90(label_pt, dims=(0, 1))
            elif aug == 4:
                label_pt = torch.rot90(label_pt, dims=(0, 1), k=2)
            elif aug == 5:
                label_pt = torch.rot90(label_pt, dims=(0, 1), k=3)
            elif aug == 6:
                label_pt = torch.rot90(label_pt.flip(0), dims=(0, 1))
            elif aug == 7:
                label_pt = torch.rot90(label_pt.flip(1), dims=(0, 1))

            if aug == 1:
                blurred_pt = blurred_pt.flip(0)
            elif aug == 2:
                blurred_pt = blurred_pt.flip(1)
            elif aug == 3:
                blurred_pt = torch.rot90(blurred_pt, dims=(0, 1))
            elif aug == 4:
                blurred_pt = torch.rot90(blurred_pt, dims=(0, 1), k=2)
            elif aug == 5:
                blurred_pt = torch.rot90(blurred_pt, dims=(0, 1), k=3)
            elif aug == 6:
                blurred_pt = torch.rot90(blurred_pt.flip(0), dims=(0, 1))
            elif aug == 7:
                blurred_pt = torch.rot90(blurred_pt.flip(1), dims=(0, 1))

            if aug == 1:
                sat_mask_pt = sat_mask_pt.flip(0)
            elif aug == 2:
                sat_mask_pt = sat_mask_pt.flip(1)
            elif aug == 3:
                sat_mask_pt = torch.rot90(sat_mask_pt, dims=(0, 1))
            elif aug == 4:
                sat_mask_pt = torch.rot90(sat_mask_pt, dims=(0, 1), k=2)
            elif aug == 5:
                sat_mask_pt = torch.rot90(sat_mask_pt, dims=(0, 1), k=3)
            elif aug == 6:
                sat_mask_pt = torch.rot90(sat_mask_pt.flip(0), dims=(0, 1))
            elif aug == 7:
                sat_mask_pt = torch.rot90(sat_mask_pt.flip(1), dims=(0, 1))

        if not self.realistic_pipeline:
            blurred = blurred_pt
            gt = label_pt

            # crop boundary of blurred images
            blurred = blurred[boundary_size // 2:-boundary_size // 2, boundary_size // 2:-boundary_size // 2, :]
            gt = gt[boundary_size // 2:-boundary_size // 2, boundary_size // 2:-boundary_size // 2, :]

            # to_tensor
            blurred = blurred.permute((2, 0, 1)).contiguous()
            gt = gt.permute((2, 0, 1)).contiguous()

            return gt, blurred

        # -------- RSBlur Pipeline -----------------------
        # -------- INVERSE ISP PROCESS -------------------
        # inverse tone mapping
        blurred_L = rgb2lin_pt(blurred_pt)

        # saturation synthesis
        alpha_saturation = random.uniform(0.25, 1.75)
        blurred_L = blurred_L + (alpha_saturation * sat_mask_pt)
        blurred_L = torch.clamp(blurred_L, 0, 1)

        blurred_sat = blurred_L.clone()

        # from linear RGB to XYZ
        img_XYZ = lin2xyz(blurred_L)  # XYZ

        # from XYZ to Cam
        img_Cam = apply_cmatrix(img_XYZ, xyz2cam_realblur)  # raw RGB

        # Mosaic
        bayer_pattern = random.choice(['RGGB', 'BGGR', 'GRBG', 'GBRG'])
        img_mosaic = mosaic_bayer(img_Cam, bayer_pattern)

        # inverse white balance
        red_gain = random.uniform(1.9, 2.4)
        blue_gain = random.uniform(1.5, 1.9)
        img_mosaic = WB_img(img_mosaic, bayer_pattern, 1 / red_gain, 1 / blue_gain)

        # -------- ADDING POISSON-GAUSSIAN NOISE ON RAW -
        # estimated noise for the RealBlur dataset
        beta1 = 8.8915e-05
        beta2 = 2.9430e-05

        img_mosaic_noise = add_Poisson_noise_random(img_mosaic, beta1, beta2)

        # -------- ISP PROCESS --------------------------
        # White balance
        img_demosaic = WB_img(img_mosaic_noise, bayer_pattern, red_gain, blue_gain)

        # demosaic
        img_demosaic = torch.nn.functional.pixel_shuffle(img_demosaic.permute(2, 0, 1).unsqueeze(0), 2)
        img_demosaic = self.demosaic.forward(img_demosaic, pattern=bayer_pattern).squeeze(0).permute(1, 2, 0)

        # from Cam to XYZ
        img_IXYZ = apply_cmatrix(img_demosaic, cam2xyz_realblur)

        # frome XYZ to linear RGB
        img_IL = xyz2lin(img_IXYZ)

        # tone mapping
        img_Irgb = lin2rgb_pt(img_IL)
        # img_Irgb = lin2rgb_a7r3_polynomial(img_IL)
        img_Irgb = torch.clamp(img_Irgb, 0, 1)  # (h, w, c)

        blurred = img_Irgb
        gt = label_pt

        # don't add noise on saturated region
        sat_region = torch.ge(blurred_sat, 1.0)
        non_sat_region = torch.logical_not(sat_region)
        blurred = (blurred_sat * sat_region) + (blurred * non_sat_region)

        # Adopt a7r3 CRF
        gt = lin2rgb_a7r3_polynomial(rgb2lin_pt(gt))
        blurred = lin2rgb_a7r3_polynomial(rgb2lin_pt(blurred))

        # crop boundary of blurred images
        blurred = blurred[boundary_size // 2:-boundary_size // 2, boundary_size // 2:-boundary_size // 2, :]
        gt = gt[boundary_size // 2:-boundary_size // 2, boundary_size // 2:-boundary_size // 2, :]

        # to_tensor
        blurred = blurred.permute((2, 0, 1)).contiguous()
        gt = gt.permute((2, 0, 1)).contiguous()

        return gt, blurred

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError

class RealBlurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False, center_crop=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'input/'))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test
        self.center_crop = center_crop

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'input', self.image_list[idx]))
        label = Image.open(os.path.join(self.image_dir, 'target', self.image_list[idx]))

        if self.center_crop:
            image = TF.center_crop(image, (512, 512))
            label = TF.center_crop(label, (512, 512))

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = TF.to_tensor(image)
            label = TF.to_tensor(label)
        if self.is_test:
            name = self.image_list[idx]
            return label, image, name
        return label, image, self.image_list[idx]

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError
