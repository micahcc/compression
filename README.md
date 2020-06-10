
## Overview
Pytorch code of our recent work [A Unified End-to-End Framework for Efficient Deep Image Compression](https://arxiv.org/abs/2002.03370).

We first release the code for [Variational image compression with a scale hyperprior](https://arxiv.org/abs/1802.01436),
we will update our code to our full implementaion of our paper.


## Content

- [Prerequisites](#prerequisites)
- [Data Preparation](#data-preparation)
- [Training](#training)

## Prerequisites

You should install the libraries of this repo.

```
pip install -r requirements.txt
```

## Data Preparation

We need to first prepare the training and validation data.
The trainging data is from flicker.com.
You can obtain the training data according to description of [CompressionData](https://github.com/liujiaheng/CompressionData).

The validation data is the popular kodak dataset.
```
bash data/download_kodak.sh
```

## Training 

For high bitrate (4096, 6144, 8192), the out_channel_N is 192 and the out_channel_M is 320 in 'config_high.json'.
For low bitrate (256, 512, 1024, 2048), the out_channel_N is 128 and the out_channel_M is 192 in 'config_low.json'.

### Details

#### PSNR experiments.

For high bitrate of 8192, we first train from scratch as follows.

```
CUDA_VISIBLE_DEVICES=0 python train.py --config examples/example/config_high.json -n baseline_8192 --train flicker_path --val kodak_path
```
For other high bitrate (4096, 6144), we use the converged model of 8192 as pretrain model and set the learning rate as 1e-5.
The training iterations are set as 500000.

The low bitrate (256, 512, 1024, 2048) training process follows the same strategy.

#### MS-SSIM experiments

You should change the distorsion loss to (1-MS_SSIM), and fine-tune the pretrained model optimized by PSNR to accelerate the training process. 
You can find more details in our released paper. The training strategy is similar.

If your find our code is helpful for your research, please cite our paper.
Besides, this code is only for research.
```
@article{liu2020unified,
  title={A Unified End-to-End Framework for Efficient Deep Image Compression},
  author={Liu, Jiaheng and Lu, Guo and Hu, Zhihao and Xu, Dong},
  journal={arXiv preprint arXiv:2002.03370},
  year={2020}
}
```
