
## Overview

Pytorch Reimplementation for [Variational image compression with a scale hyperprior](https://arxiv.org/abs/1802.01436)


## Content

- [Prerequisites](#prerequisites)
- [Data Preparation](#data-preparation)
- [Training](#training)

## Prerequisites

You should install the libraries of this repo.
``
pip install -r requirements.txt
``
## Data Preparation

We need to first prepare the training and validation data.
The trainging data is from flicker.com.
You can download the data in [link](https://bhpan.buaa.edu.cn:443/link/5A49A835629D57929F82A2D97A6EC855).
Then you can generate the 256*256 patches using following script.
```
python data/generatePatchFlickr.py flicker_2W_images/ output_path/
```
The size of output training dataset is over 80G.

The validation data is the popular kodak dataset.
```
bash data/download_kodak.sh
```

## Training 

For high bitrate (4096, 6144, 8192), the out_channel_N is 192 and the out_channel_M is 320 in 'config_high.json'.
For low bitrate (256, 512, 1024, 2048), the out_channel_N is 128 and the out_channel_M is 192 in 'config_low.json'.

### Details
For high bitrate of 8192, we first train from scratch as follows.

```
CUDA_VISIBLE_DEVICES=0 python train.py --config examples/example/config_high.json -n baseline_8192 --train flicker_path --val kodak_path
```
For other high bitrate (4096, 6144), we use the converged model of 8192 as pretrain model and set the learning rate as 1e-5.
The training iterations is set as 50w.

The low bitrate training process follows the same strategy.

If your find our code is helpful for your research, please cite our paper.
Besides, this code is only for research.
```
@article{liu2020unified,
  title={A Unified End-to-End Framework for Efficient Deep Image Compression},
  author={Liu, Jiaheng and Lu, Guo and Hu, Zhihao and Xu, Dong},
  journal={arXiv preprint arXiv:2002.03370},
  year={2020}
}