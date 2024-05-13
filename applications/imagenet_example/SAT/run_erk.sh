#!/bin/bash

# cmd: sh run.sh

msb=/home/yongyang/work/projects/msb/sparsity

export PYTHONPATH=$msb:$PYTHONPATH

python main.py -a resnet18 --pretrained --sparse --sparse_alloc ERK /home/yongyang/data/datasets/imagenet
