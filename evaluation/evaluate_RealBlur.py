## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808

import os
import numpy as np
from glob import glob
from natsort import natsorted
from skimage import io
import cv2
from skimage.metrics import structural_similarity
from tqdm import tqdm
import concurrent.futures
import argparse

parser = argparse.ArgumentParser(description='eval arg')
parser.add_argument('--input_dir', type=str, default='')
parser.add_argument('--out_txt', type=str, default='')
parser.add_argument('--gt_root', type=str, default='RealBlurJ_test')
parser.add_argument('--core', type=int, default=8)
args = parser.parse_args()

def image_align(deblurred, gt):
  # this function is based on kohler evaluation code
  z = deblurred
  c = np.ones_like(z)
  x = gt

  zs = (np.sum(x * z) / np.sum(z * z)) * z # simple intensity matching

  warp_mode = cv2.MOTION_HOMOGRAPHY
  warp_matrix = np.eye(3, 3, dtype=np.float32)

  # Specify the number of iterations.
  number_of_iterations = 100

  termination_eps = 0

  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
              number_of_iterations, termination_eps)

  # Run the ECC algorithm. The results are stored in warp_matrix.
  (cc, warp_matrix) = cv2.findTransformECC(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.cvtColor(zs, cv2.COLOR_RGB2GRAY), warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=5)

  target_shape = x.shape
  shift = warp_matrix

  zr = cv2.warpPerspective(
    zs,
    warp_matrix,
    (target_shape[1], target_shape[0]),
    flags=cv2.INTER_CUBIC+ cv2.WARP_INVERSE_MAP,
    borderMode=cv2.BORDER_REFLECT)

  cr = cv2.warpPerspective(
    np.ones_like(zs, dtype='float32'),
    warp_matrix,
    (target_shape[1], target_shape[0]),
    flags=cv2.INTER_NEAREST+ cv2.WARP_INVERSE_MAP,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=0)

  zr = zr * cr
  xr = x * cr

  return zr, xr, cr, shift

def compute_psnr(image_true, image_test, image_mask, data_range=None):
  # this function is based on skimage.metrics.peak_signal_noise_ratio
  err = np.sum((image_true - image_test) ** 2, dtype=np.float64) / np.sum(image_mask)
  return 10 * np.log10((data_range ** 2) / err)


def compute_ssim(tar_img, prd_img, cr1):
    ssim_pre, ssim_map = structural_similarity(tar_img, prd_img, multichannel=True, gaussian_weights=True, use_sample_covariance=False, data_range = 1.0, full=True)
    ssim_map = ssim_map * cr1
    r = int(3.5 * 1.5 + 0.5)  # radius as in ndimage
    win_size = 2 * r + 1
    pad = (win_size - 1) // 2
    ssim = ssim_map[pad:-pad,pad:-pad,:]
    crop_cr1 = cr1[pad:-pad,pad:-pad,:]
    ssim = ssim.sum(axis=0).sum(axis=0)/crop_cr1.sum(axis=0).sum(axis=0)
    ssim = np.mean(ssim)
    return ssim

def proc(filename):
    tar,prd = filename
    tar_img = io.imread(tar)
    prd_img = io.imread(prd)
    
    tar_img = tar_img.astype(np.float32)/255.0
    prd_img = prd_img.astype(np.float32)/255.0
    
    prd_img, tar_img, cr1, shift = image_align(prd_img, tar_img)

    prd_img = np.clip(prd_img, 0, 1)
    tar_img = np.clip(tar_img, 0, 1)

    PSNR = compute_psnr(tar_img, prd_img, cr1, data_range=1)
    SSIM = compute_ssim(tar_img, prd_img, cr1)
    return (PSNR,SSIM)

if __name__ == '__main__':

  input_dir = args.input_dir
  if args.out_txt == '':
    out_txt = input_dir.split('/')[-2] + '.txt'
  else:
    out_txt = args.out_txt
  print(out_txt)

  file_path = os.path.join(input_dir)

  path_list = natsorted(glob(os.path.join(file_path, '*.png')) + glob(os.path.join(file_path, '*.jpg')))
  print(len(path_list))

  gt_list = []
  for inp_path in path_list:
    inp_path_split = inp_path.split('/')[-1].split('_')
    scene_name = inp_path_split[0]
    img_name = '_'.join(inp_path_split[1:])

    gt_path = os.path.join(args.gt_root, scene_name, 'blur', img_name)
    gt_list.append(gt_path)


  psnr, ssim, files = [], [], []
  img_files =[(i, j) for i,j in zip(gt_list,path_list)]
  with concurrent.futures.ProcessPoolExecutor(max_workers=args.core) as executor:
      for filename, PSNR_SSIM in zip(img_files, executor.map(proc, img_files)):
          psnr.append(PSNR_SSIM[0])
          ssim.append(PSNR_SSIM[1])
          files.append(filename[0])

  avg_psnr = sum(psnr)/len(psnr)
  avg_ssim = sum(ssim)/len(ssim)

  print('For {:s} dataset PSNR: {:f} SSIM: {:f}\n'.format(input_dir, avg_psnr, avg_ssim))

  txt_list = []
  for i in range(len(psnr)):
    txt = '{:s} {:f} {:f}\n'.format(files[i], psnr[i], ssim[i])
    txt_list.append(txt)

  txt_list.append('For {:s} dataset PSNR: {:f} SSIM: {:f}\n'.format(input_dir, avg_psnr, avg_ssim))

  with open(out_txt, 'wt') as f:
      f.writelines(txt_list)
