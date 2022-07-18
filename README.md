# Realistic Blur Synthesis for Learning Image Deblurring
by Jaesung Rim, Geonung Kim, Jungeon Kim, [Junyong Lee](https://junyonglee.me/), [Seungyong Lee](http://cg.postech.ac.kr/leesy/), [Sunghyun Cho](https://www.scho.pe.kr/). [[pdf]](http://cg.postech.ac.kr/research/RealBlur/assets/pdf/RealBlur_eccv2020.pdf) [[project]](http://cg.postech.ac.kr/research/RealBlur/)

### Deblurring Results
<img src="./imgs/qualatitive_results.png" width="100%" alt="Real Photo">

## Installation 

```bash
git clone --recurse-submodules https://github.com/rimchang/RSBlur.git
```

## Tested environment

We recommend a virtual environment using conda or docker.

##### SRN-Deblur
- Tensorflow 1.15
- cuda11.1

## Download

For testing, download [RealBlur](https://cgdata.postech.ac.kr/sharing/YhKdbtvD0).

For training same as our paper, download [RealBlur](https://cgdata.postech.ac.kr/sharing/YhKdbtvD0), [BSD-B](https://cgdata.postech.ac.kr/sharing/ak2v58DFR), [GoPro](https://cv.snu.ac.kr/~snah/Deblur/dataset/GOPRO_Large.zip).

All datasets should be located SRN-Deblur/testing_set/, SRN-Deblur/training_set/, DeblurGAN-v2/dataset/. 

Also, we provide [trained model](https://cgdata.postech.ac.kr/sharing/arLpxqXvT). Please move checkpoint files to SRN-Deblur/checkpoints, DeblurGAN-v2/checkpoints.

Please check "link_file.sh" for appropriate linking of directories and files.

If you have network problem, please use [google drive link](https://drive.google.com/drive/folders/1xUNAAVzLhNQuGriKTk1hrE-MT-H_56fq).


## Training

```bash
# ./SRN-Deblur-RSBlur

# RSBlur
python run_model.py --phase=train --checkpoint_path=0719_SRN-Deblur_RSBlur_real --sat_synthesis=None --noise_synthesis=None --datalist=../datalist/RSBlur/RSBlur_real_train.txt --gpu=0
python run_model.py --phase=train --checkpoint_path=0719_SRN-Deblur_RSBlur_syn --sat_synthesis=None --noise_synthesis=None --datalist=../datalist/RSBlur/RSBlur_syn_train.txt --gpu=0
python run_model.py --phase=train --checkpoint_path=0719_SRN-Deblur_RSBlur_syn_with_ours --sat_synthesis=sat_synthesis --noise_synthesis=poisson_RSBlur --cam_params_RSBlur=1 --datalist=../datalist/RSBlur/RSBlur_syn_train.txt --gpu=0

# GoPro
python run_model.py --phase=train --checkpoint_path=0719_SRN-Deblur_GoPro_ABME_with_ours --sat_synthesis=sat_synthesis --noise_synthesis=poisson_gamma --cam_params_RealBlur=1 --adopt_crf_realblur=1 --datalist=../datalist/GoPro/GoPro_INTER_ABME_train.txt --gpu=0
python run_model.py --phase=train --checkpoint_path=0719_SRN-Deblur_U_with_ours --sat_synthesis=sat_synthesis --noise_synthesis=poisson_gamma --cam_params_RealBlur=1 --adopt_crf_realblur=1 --datalist=../datalist/GoPro/GoPro_U_train.txt --gpu=0

# RealBlur
python run_model.py --phase=train --checkpoint_path=0719_SRN-Deblur_Realblur_j --sat_synthesis=None --noise_synthesis=None --datalist=../datalist/RealBlur_J_train_list.txt --gpu=0
```

## Testing

```bash
# ./SRN-Deblur-RSBlur

# RSBlur
python run_model.py --phase=test --checkpoint_path=SRN-Deblur_RSBlur_real --datalist=../datalist/RSBlur/RSBlur_real_test.txt --gpu=0
python run_model.py --phase=test --checkpoint_path=SRN-Deblur_RSBlur_syn --datalist=../datalist/RSBlur/RSBlur_real_test.txt --gpu=0
python run_model.py --phase=test --checkpoint_path=SRN-Deblur_RSBlur_syn_with_ours --datalist=../datalist/RSBlur/RSBlur_real_test.txt --gpu=0

# RealBlur
python run_model.py --phase=test --checkpoint_path=SRN-Deblur_GoPro_ABME_with_ours --datalist=../datalist/RealBlur_J_test_list.txt --gpu=0
python run_model.py --phase=test --checkpoint_path=SRN-Deblur_U_with_ours --datalist=../datalist/RealBlur_J_test_list.txt --gpu=0
python run_model.py --phase=test --checkpoint_path=SRN-Deblur_Realblur_j --datalist=../datalist/RealBlur_J_test_list.txt --gpu=0```
```

## Evaluation

```bash
# ./evaluation

python evaluate_RSBlur.py --input_dir=../SRN-Deblur-RSBlur/testing_res/SRN-Deblur_RSBlur_real --gt_root=../SRN-Deblur-RSBlur/dataset/RSBlur;
python evaluate_RealBlur.py --input_dir=../SRN-Deblur-RSBlur/testing_res/SRN-Deblur_U_with_ours --gt_root=../SRN-Deblur-RSBlur/RealBlur-J_ECC_IMCORR_centroid_itensity_ref;
```

## License

The RSBlur dataset is released under CC BY 4.0 license.

## Acknowledment

The code is based on the [SRN-Deblur](https://github.com/jiangsutx/SRN-Deblur), [CBDNet](https://github.com/GuoShi28/CBDNet) and [UID](https://github.com/timothybrooks/unprocessing).

## Citation

If you use our dataset for your research, please cite our papers.

```bibtex
@inproceedings{rim_2022_ECCV,
 title={Realistic Blur Synthesis for Learning Image Deblurring},
 author={Jaesung Rim, Geonung Kim, Jungeon Kim, Junyong Lee, Seungyong Lee, Sunghyun Cho},
 booktitle={Proceedings of the European Conference on Computer Vision (ECCV)}
 year={2022}
}
```
