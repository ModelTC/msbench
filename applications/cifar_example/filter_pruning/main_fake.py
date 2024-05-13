# 訓練過程參考
# https://github.com/Eclipsess/CHIP_NeurIPS2021
# https://github.com/lmbxmu/HRankPlus

# 運行腳本
# PYTHONPATH=/path/sparsity-master python main_real.py --data_dir ./data --result_dir ./result/resnet_110/real/norm/ --arch resnet_110  --batch_size 256 --epochs 400 --lr_type cos --learning_rate 0.01 --momentum 0.9 --weight_decay 0.005 --pretrain_dir pretrain_dir/resnet_110.pt --sparsity [0.]+[0.2]*2+[0.3]*18+[0.35]*36 --gpu 0 --mask_type norm --func p1
# PYTHONPATH=/path/sparsity-master python main_real.py --data_dir ./data --result_dir ./result/resnet_110/real/nuclear/ --arch resnet_110  --batch_size 256 --epochs 400 --lr_type cos --learning_rate 0.01 --momentum 0.9 --weight_decay 0.005 --pretrain_dir pretrain_dir/resnet_110.pt --sparsity [0.]+[0.2]*2+[0.3]*18+[0.35]*36 --gpu 1 --mask_type nuclear

import os
import numpy as np
import time, datetime
import argparse
import copy
from thop import profile
from collections import OrderedDict

import math
import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
from models import resnet_56, resnet_110
from msbench.scheduler import build_sparse_scheduler
from msbench.gen_sparse_rate import *
import torch
import torch.utils
import torch.utils.data.distributed

import torchvision
from torchvision import datasets, transforms
import utils
import re
from export import *

parser = argparse.ArgumentParser("CIFAR-10 training")

parser.add_argument("--data_dir", type=str, default="./data", help="path to dataset")

parser.add_argument(
    "--arch",
    type=str,
    default="resnet_56",
    choices=("resnet_56", "resnet_110"),
    help="architecture to calculate feature maps",
)

parser.add_argument("--lr_type", type=str, default="cos", help="lr type")

parser.add_argument(
    "--result_dir",
    type=str,
    default="./result",
    help="results path for saving models and loggers",
)

parser.add_argument("--batch_size", type=int, default=256, help="batch size")

parser.add_argument("--epochs", type=int, default=200, help="num of training epochs")

parser.add_argument("--label_smooth", type=float, default=0, help="label smoothing")

parser.add_argument(
    "--learning_rate", type=float, default=0.01, help="init learning rate"
)

parser.add_argument("--lr_decay_step", default="50,100", type=str, help="learning rate")

parser.add_argument("--momentum", type=float, default=0.9, help="momentum")

parser.add_argument("--weight_decay", type=float, default=0.005, help="weight decay")

parser.add_argument("--pretrain_dir", type=str, default="", help="pretrain model path")

parser.add_argument("--ci_dir", type=str, default="", help="ci path")

parser.add_argument("--sparse_rate", type=float, default=0.5)

parser.add_argument("--gpu", type=str, default="0", help="gpu id")

parser.add_argument("--func", default="p1", choices=["p1", "p2", "l1", "l2", "cos"])

parser.add_argument("--mask_type", default="norm", choices=["fpgm", "norm"])

MaskGeneratorDict = {"fpgm": "FPGMMaskGenerator", "norm": "NormalMaskGenerator"}


args = parser.parse_args()
CLASSES = 10

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
CLASSES = 10
print_freq = (256 * 50) // args.batch_size

if not os.path.isdir(args.result_dir):
    os.makedirs(args.result_dir)

utils.record_config(args)
now = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
logger = utils.get_logger(os.path.join(args.result_dir, "logger" + now + ".log"))


def load_cifar_data(args):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    trainset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=transform_train
    )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=1
    )
    testset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=transform_test
    )
    val_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=1
    )

    return train_loader, val_loader


def adjust_learning_rate(optimizer, epoch, step, len_iter):
    if args.lr_type == "step":
        factor = epoch // 125
        # if epoch >= 80:
        #     factor = factor + 1
        lr = args.learning_rate * (0.1**factor)

    elif args.lr_type == "step_5":
        factor = epoch // 10
        if epoch >= 80:
            factor = factor + 1
        lr = args.learning_rate * (0.5**factor)

    elif args.lr_type == "cos":  # cos without warm-up
        lr = (
            0.5
            * args.learning_rate
            * (1 + math.cos(math.pi * (epoch - 5) / (args.epochs - 5)))
        )

    elif args.lr_type == "exp":
        step = 1
        decay = 0.96
        lr = args.learning_rate * (decay ** (epoch // step))

    elif args.lr_type == "fixed":
        lr = args.learning_rate
    else:
        raise NotImplementedError

    # Warmup
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_iter) / (5.0 * len_iter)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if step == 0:
        logger.info("learning_rate: " + str(lr))


def main():
    cudnn.benchmark = True
    cudnn.enabled = True
    logger.info("args = %s", args)

    # load large model
    logger.info("resuming from pretrain model")
    origin_model = eval(args.arch)(sparsity=[0.0] * 100).cuda()
    ckpt = torch.load(args.pretrain_dir, map_location="cuda:0")

    if args.arch == "resnet_110":
        new_state_dict = OrderedDict()
        for k, v in ckpt["state_dict"].items():
            new_state_dict[k.replace("module.", "")] = v
        origin_model.load_state_dict(new_state_dict)
    else:
        origin_model.load_state_dict(ckpt["state_dict"])

    # calculate model size
    input_image_size = 32
    input_image = torch.randn(1, 3, input_image_size, input_image_size).cuda()
    flops, params = profile(origin_model, inputs=(input_image,))
    logger.info("Origin model")
    logger.info("Params: %.2f" % (params))
    logger.info("Flops: %.2f" % (flops))

    prepare_custom_config_dict = {
        "scheduler": {"type": "BaseScheduler"},
        "mask_generator": {
            "type": MaskGeneratorDict[args.mask_type],
            "kwargs": {"structured": True, "func": args.func},
        },
        "fake_sparse": {"type": "DefaultFakeSparse"},
    }

    logger.info(f"Mask config: {prepare_custom_config_dict}")

    sparse_scheduler = build_sparse_scheduler(prepare_custom_config_dict)
    origin_model = sparse_scheduler.prepare_sparse_model(origin_model)

    sparsities = get_st_sparsities_uniform(origin_model, args.sparse_rate)

    update_sparsity_per_layer_from_sparsities(origin_model, sparsities)

    logger.info(f"sparsities {sparsities}")

    # load training data
    train_loader, val_loader = load_cifar_data(args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = utils.CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    if len(args.gpu) > 1:
        device_id = []
        for i in range((len(args.gpu) + 1) // 2):
            device_id.append(i)
        origin_model = nn.DataParallel(origin_model, device_ids=device_id).cuda()

    optimizer = torch.optim.SGD(
        origin_model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    lr_decay_step = list(map(int, args.lr_decay_step.split(",")))

    start_epoch = 0
    best_top1_acc = 0

    # load the checkpoint if it exists
    checkpoint_dir = os.path.join(args.result_dir, "checkpoint.pth.tar")

    # train the model
    epoch = start_epoch
    while epoch < args.epochs:
        if args.label_smooth > 0:
            train_obj, train_top1_acc, train_top5_acc = train(
                epoch, train_loader, origin_model, criterion_smooth, optimizer
            )  # , scheduler)
        else:
            train_obj, train_top1_acc, train_top5_acc = train(
                epoch, train_loader, origin_model, criterion, optimizer
            )  # , scheduler)
        valid_obj, valid_top1_acc, valid_top5_acc = validate(
            epoch, val_loader, origin_model, criterion, args
        )

        is_best = False
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            is_best = True

        sparse_scheduler.export_sparse_model(origin_model)
        utils.save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": origin_model.state_dict(),
                "best_top1_acc": best_top1_acc,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            args.result_dir,
        )

        epoch += 1
        logger.info("=>Best accuracy {:.3f}".format(best_top1_acc))  #


def train(epoch, train_loader, model, criterion, optimizer, scheduler=None):
    batch_time = utils.AverageMeter("Time", ":6.3f")
    data_time = utils.AverageMeter("Data", ":6.3f")
    losses = utils.AverageMeter("Loss", ":.4e")
    top1 = utils.AverageMeter("Acc@1", ":6.2f")
    top5 = utils.AverageMeter("Acc@5", ":6.2f")

    model.train()
    end = time.time()

    for param_group in optimizer.param_groups:
        cur_lr = param_group["lr"]
    logger.info("learning_rate: " + str(cur_lr))

    num_iter = len(train_loader)
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda()
        target = target.cuda()

        adjust_learning_rate(optimizer, epoch, i, num_iter)

        # compute outputy
        logits = model(images)
        loss = criterion(logits, target)

        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)  # accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logger.info(
                "Epoch[{0}]({1}/{2}): "
                "Loss {loss.avg:.4f} "
                "Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}".format(
                    epoch, i, num_iter, loss=losses, top1=top1, top5=top5
                )
            )

    # scheduler.step()

    return losses.avg, top1.avg, top5.avg


def validate(epoch, val_loader, model, criterion, args):
    batch_time = utils.AverageMeter("Time", ":6.3f")
    losses = utils.AverageMeter("Loss", ":.4e")
    top1 = utils.AverageMeter("Acc@1", ":6.2f")
    top5 = utils.AverageMeter("Acc@5", ":6.2f")

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            logits = model(images)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            pred1, pred5 = utils.accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        logger.info(
            " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
        )

    return losses.avg, top1.avg, top5.avg


if __name__ == "__main__":
    main()
