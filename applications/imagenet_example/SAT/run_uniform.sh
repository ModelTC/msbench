#!/bin/bash

# cmd: sh run.sh

msb=/path to msbench file folder

export PYTHONPATH=$msb:$PYTHONPATH

python main.py -a resnet18 --pretrained --sparse --sparse_alloc uniform /home/yongyang/data/datasets/imagenet
