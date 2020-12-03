#!/usr/bin/env python3

import os
import argparse
import json
import time
import logging
import sys

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.ms_ssim_torch import ms_ssim
from model import save_model, ImageCompressor, load_model
from datasets import Dataset
from tensorboardX import SummaryWriter
from Meter import AverageMeter

torch.backends.cudnn.enabled = True

logger = logging.getLogger("ImageCompression")


class Config:
    def __init__(self):
        self.gpu_num = 4
        self.gpu_num = torch.cuda.device_count()
        self.base_lr = 1e-4
        self.train_lambda = 8192
        self.print_freq = 100
        self.warmup_step = 0
        self.batch_size = 4
        self.tot_epoch = 1000000
        self.tot_step = 2500000
        self.decay_interval = 2200000
        self.lr_decay = 0.1
        self.image_size = 256
        self.save_model_freq = 50000
        self.test_step = 10000
        self.out_channel_N = 192
        self.out_channel_M = 320
        self.in_channel = 1

    def parse_config(self, configFile):
        with open(configFile) as f:
            config = json.load(f)
        if "tot_epoch" in config:
            self.tot_epoch = config["tot_epoch"]
        if "tot_step" in config:
            self.tot_step = config["tot_step"]
        if "train_lambda" in config:
            self.train_lambda = config["train_lambda"]
            if self.train_lambda < 4096:
                self.out_channel_N = 128
                self.out_channel_M = 192
            else:
                self.out_channel_N = 192
                self.out_channel_M = 320
        if "batch_size" in config:
            self.batch_size = config["batch_size"]
        if "print_freq" in config:
            self.print_freq = config["print_freq"]
        if "test_step" in config:
            self.test_step = config["test_step"]
        if "save_model_freq" in config:
            self.save_model_freq = config["save_model_freq"]
        if "lr" in config:
            if "base" in config["lr"]:
                self.base_lr = config["lr"]["base"]
            if "decay" in config["lr"]:
                self.lr_decay = config["lr"]["decay"]
            if "decay_interval" in config["lr"]:
                self.decay_interval = config["lr"]["decay_interval"]
        if "out_channel_N" in config:
            self.out_channel_N = config["out_channel_N"]
        if "out_channel_M" in config:
            self.out_channel_M = config["out_channel_M"]


def adjust_learning_rate(config, optimizer, global_step):
    if global_step < config.warmup_step:
        lr = config.base_lr * global_step / config.warmup_step
    elif global_step < config.decay_interval:
        lr = config.base_lr
    else:
        # lr = base_lr * (lr_decay ** (global_step // decay_interval))
        lr = config.base_lr * config.lr_decay
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr


def train(net, train_loader, config, tb_logger, epoch, global_step, cur_lr, optimizer):
    logger.info("Epoch {} begin".format(epoch))
    net.train()

    elapsed, losses, psnrs, bpps, bpp_features, bpp_zs, mse_losses = [
        AverageMeter(config.print_freq) for _ in range(7)
    ]
    # model_time = 0
    # compute_time = 0
    # log_time = 0
    for batch_idx, input in enumerate(train_loader):
        start_time = time.time()
        global_step += 1
        clipped_recon_image, mse_loss, bpp_feature, bpp_z, bpp = net(input)
        distribution_loss = bpp
        distortion = mse_loss
        rd_loss = config.train_lambda * distortion + distribution_loss
        optimizer.zero_grad()
        rd_loss.backward()

        def clip_gradient(optimizer, grad_clip):
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)

        clip_gradient(optimizer, 5)
        optimizer.step()
        # model_time += (time.time()-start_time)
        if (global_step % config.print_freq) == 0:
            # t0 = time.time()
            if mse_loss.item() > 0:
                psnr = 10 * (torch.log(1 * 1 / mse_loss) / np.log(10))
                psnrs.update(psnr.item())
            else:
                psnrs.update(100)
            # t1 = time.time()
            elapsed.update(time.time() - start_time)
            losses.update(rd_loss.item())
            bpps.update(bpp.item())
            bpp_features.update(bpp_feature.item())
            bpp_zs.update(bpp_z.item())
            mse_losses.update(mse_loss.item())

            # begin = time.time()
            tb_logger.add_scalar("lr", cur_lr, global_step)
            tb_logger.add_scalar("rd_loss", losses.avg, global_step)
            tb_logger.add_scalar("psnr", psnrs.avg, global_step)
            tb_logger.add_scalar("bpp", bpps.avg, global_step)
            tb_logger.add_scalar("bpp_feature", bpp_features.avg, global_step)
            tb_logger.add_scalar("bpp_z", bpp_zs.avg, global_step)
            process = global_step / config.tot_step * 100.0
            log = " | ".join(
                [
                    f"Step [{global_step}/{config.tot_step}={process:.2f}%]",
                    f"Epoch {epoch}",
                    f"Time {elapsed.val:.3f} ({elapsed.avg:.3f})",
                    f"Lr {cur_lr}",
                    f"Total Loss {losses.val:.3f} ({losses.avg:.3f})",
                    f"PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})",
                    f"Bpp {bpps.val:.5f} ({bpps.avg:.5f})",
                    f"Bpp_feature {bpp_features.val:.5f} ({bpp_features.avg:.5f})",
                    f"Bpp_z {bpp_zs.val:.5f} ({bpp_zs.avg:.5f})",
                    f"MSE {mse_losses.val:.5f} ({mse_losses.avg:.5f})",
                ]
            )
            logger.info(log)

        if (global_step % config.save_model_freq) == 0:
            save_model(net, global_step, config.save_path)
        if (global_step % config.test_step) == 0:
            testKodak(global_step)
            net.train()

    return global_step


def testKodak(net, test_loader, step, tb_logger):
    with torch.no_grad():
        net.eval()
        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0
        cnt = 0
        for batch_idx, input in enumerate(test_loader):
            clipped_recon_image, mse_loss, bpp_feature, bpp_z, bpp = net(input)
            mse_loss, bpp_feature, bpp_z, bpp = (
                torch.mean(mse_loss),
                torch.mean(bpp_feature),
                torch.mean(bpp_z),
                torch.mean(bpp),
            )
            psnr = 10 * (torch.log(1.0 / mse_loss) / np.log(10))
            sumBpp += bpp
            sumPsnr += psnr
            msssim = ms_ssim(
                clipped_recon_image.cpu().detach(),
                input,
                data_range=1.0,
                size_average=True,
            )
            msssimDB = -10 * (torch.log(1 - msssim) / np.log(10))
            sumMsssimDB += msssimDB
            sumMsssim += msssim
            logger.info(
                "Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(
                    bpp, psnr, msssim, msssimDB
                )
            )
            cnt += 1

        logger.info("Test on Kodak dataset: model-{}".format(step))
        sumBpp /= cnt
        sumPsnr /= cnt
        sumMsssim /= cnt
        sumMsssimDB /= cnt
        logger.info(
            "Dataset Average result---Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(
                sumBpp, sumPsnr, sumMsssim, sumMsssimDB
            )
        )
        if tb_logger is not None:
            logger.info("Add tensorboard---Step:{}".format(step))
            tb_logger.add_scalar("BPP_Test", sumBpp, step)
            tb_logger.add_scalar("PSNR_Test", sumPsnr, step)
            tb_logger.add_scalar("MS-SSIM_Test", sumMsssim, step)
            tb_logger.add_scalar("MS-SSIM_DB_Test", sumMsssimDB, step)
        else:
            logger.info("No need to add tensorboard")


def setup_logger(name, save_path):
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s] %(message)s")
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][L%(lineno)d][%(levelname)s] %(message)s"
    )
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    logger.setLevel(logging.INFO)
    if name != "":
        os.makedirs(save_path, exist_ok=True)
        filehandler = logging.FileHandler(os.path.join(save_path, "log.txt"))
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)


def main():
    parser = argparse.ArgumentParser(
        description="Pytorch reimplement for variational image compression with a scale hyperprior"
    )

    parser.add_argument("-n", "--name", default="", help="experiment name")
    parser.add_argument("-p", "--pretrain", default="", help="load pretrain model")
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--config", dest="config", required=False, help="hyperparameter in json format"
    )
    parser.add_argument(
        "--seed",
        default=234,
        type=int,
        help="seed for random functions, and network initialization",
    )
    parser.add_argument(
        "--train", dest="train", required=True, help="the path of training dataset"
    )
    parser.add_argument(
        "--val", dest="val", required=True, help="the path of validation dataset"
    )

    args = parser.parse_args()

    torch.manual_seed(seed=args.seed)
    config = Config()
    if args.config:
        config.parse_config(args.config)
    save_path = os.path.join("checkpoints", args.name)
    setup_logger(args.name, save_path)

    tb_logger = None
    logger.info("image compression training")
    logger.info(
        f"out_channel_N:{config.out_channel_N}, out_channel_M:{config.out_channel_M}"
    )

    model = ImageCompressor(
        config.in_channel, config.out_channel_N, config.out_channel_M
    )
    if args.pretrain != "":
        logger.info("loading model:{}".format(args.pretrain))
        global_step = load_model(model, args.pretrain)
    else:
        global_step = 0

    net = model.cuda()
    net = torch.nn.DataParallel(net, list(range(config.gpu_num)))
    parameters = net.parameters()
    test_dataset = Dataset(data_dir=args.val, augment=False, channels=1, depth=16)
    test_loader = DataLoader(
        dataset=test_dataset,
        shuffle=False,
        batch_size=1,
        pin_memory=True,
        num_workers=1,
    )
    optimizer = optim.Adam(parameters, lr=config.base_lr)
    # save_model(model, 0)
    tb_logger = SummaryWriter(os.path.join(save_path, "events"))
    train_data_dir = args.train
    train_dataset = Dataset(train_data_dir, config.image_size, channels=1, depth=16)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )

    if args.test:
        testKodak(net, test_loader, global_step, tb_logger)
        return -1

    steps_epoch = global_step // (len(train_dataset) // (config.batch_size))
    save_model(model, global_step, save_path)
    for epoch in range(steps_epoch, config.tot_epoch):
        cur_lr = adjust_learning_rate(config, optimizer, global_step)
        if global_step > config.tot_step:
            save_model(model, global_step, save_path)
            break
        global_step = train(
            net=net,
            train_loader=train_loader,
            config=config,
            tb_logger=tb_logger,
            epoch=epoch,
            global_step=global_step,
            cur_lr=cur_lr,
            optimizer=optimizer,
        )
        save_model(model, global_step, save_path)


if __name__ == "__main__":
    sys.exit(main())
