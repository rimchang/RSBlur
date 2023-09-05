## Realistic Blur Synthesis for Learning Image Deblurring 
##### [Project](http://cg.postech.ac.kr/research/rsblur/) | [Paper](http://cg.postech.ac.kr/research/rsblur/assets/pdf/RSBlur.pdf) | [Supple](http://cg.postech.ac.kr/research/rsblur/assets/pdf/RSBlur-supp.pdf)

### Pytorch Implementation of ECCV Paper 

> Realistic Blur Synthesis for Learning Image Deblurring<br>
> Jaesung Rim, Geonung Kim, Jungeon Kim, [Junyong Lee](https://junyonglee.me/), [Seungyong Lee](http://cg.postech.ac.kr/leesy/), [Sunghyun Cho](https://www.scho.pe.kr/). <br>
> POSTECH<br>
> *IEEE European Conference on Computer Vision (**ECCV**) 2022*<br>

### Results of the Uformer.

| Models | Train set | Realistic Pipeline | PSNR / SSIM    |
| :---:|:---:  |  :---:|:---:|
| Uformer-B |   GoPro |  ✓   | 30.98 / 0.9067 |
| Uformer-B |  GoPro  |     | 29.08 / 0.8754 |
| Uformer-B | GoPro_U |   ✓  | 31.19 / 0.9143 |
| Uformer-B | GoPro_U |     | 28.93 / 0.8673 |

## Tested environment

We recommend a virtual environment using conda or docker.

##### Uformer
- Pytorch 1.9.0
- cuda11.1

### Pre-trained models [[Google Drive]](https://drive.google.com/drive/folders/1JcYNvIKflIaSxbD2Jn98FzGcWyAvWxoW?usp=drive_link)
<details>
<summary><strong>Descriptions</strong> (click) </summary>

- Uformer_B_RealisticGoProABMEDeblur.pth : Trained on GoPro_INTER_ABME with our synthesis pipeline.
- Uformer_B_NaiveGoProABMEDeblur.pth : Trained on GoPro_INTER_ABME in the Naive way.
- Uformer_B_RealisticGoProUDeblur.pth : Trained on GoPro_U with our synthesis pipeline.
- Uformer_B_NaiveGoProUDeblur.pth : Trained on GoPro_U in the Naive way.
</details>

## RSBlur pipeline

We provide Dataset modules for adopting our pipline.
Please check the below codes.

```python
# ./Uformer-RSBlur/dataset/dataset_RealisticDeblur.py

class RealisticGoProABMEDataset(Dataset):
    def __init__(self, image_dir, patch_size=256, image_aug=True, realistic_pipeline=True):
        ...

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        ...

class RealisticGoProUataset(Dataset):
    def __init__(self, image_dir, patch_size=256, image_aug=True, realistic_pipeline=True):
        ...

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        ...
```

## Training

```bash
# ./Uformer-RSBlur
# All datasets should be located in Uformer-RSBlur/datasets
# require two of 3090, 4~5 days

# GoPro_INTER_ABME with our pipeline
python3 train/train_RealisticGoProABMEDeblur.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
 --train_ps 256 --train_dir datasets/GOPRO_INTER_ABME \
 --val_ps 256 --val_dir datasets/RealBlurJ_test --env _RealisticGoProABMEDeblur \
 --mode deblur --nepoch 1500 --checkpoint 100 --dataset GoPro --warmup --train_workers 12

# GoPro_U with our pipeline
python3 train/train_RealisticGoProUDeblur.py --arch Uformer_B --batch_size 8 --gpu '0,1' \
 --train_ps 256 --train_dir datasets/GOPRO_INTER_ABME \
 --val_ps 256 --val_dir datasets/RealBlurJ_test --env _RealisticGoProUDeblur \
 --mode deblur --nepoch 1500 --checkpoint 100 --dataset GoPro --warmup --train_workers 12

```

## Testing

```bash
# ./Uformer-RSBlur
# All datasets should be located in Uformer-RSBlur/datasets

# Test on the RealBlur
python3 test/test_realblur_reflect.py --input_dir ./datasets/ --result_dir ./results/Uformer_B_RealisticGoProUDeblur/ --weights ./logs/Uformer_B_RealisticGoProUDeblur.pth;

# Test on the RealBlur
python3 test/test_realblur_reflect.py --input_dir ./datasets/ --result_dir ./results/Uformer_B_RealisticGoProABMEDeblur_mark9/ --weights ./logs/Uformer_B_RealisticGoProABMEDeblur.pth;

```

## License

The RSBlur dataset is released under CC BY 4.0 license.

## Citation

If you use our dataset for your research, please cite our paper.

```bibtex
@inproceedings{rim_2022_ECCV,
 title={Realistic Blur Synthesis for Learning Image Deblurring},
 author={Jaesung Rim, Geonung Kim, Jungeon Kim, Junyong Lee, Seungyong Lee, Sunghyun Cho},
 booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
 year={2022}
}
```
