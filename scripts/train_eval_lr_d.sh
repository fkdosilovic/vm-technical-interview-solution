#!/usr/bin/bash

python experiments/train_eval_lr.py --dataset /home/filip/Data/mnist --lr 0.01 --weight-decay 0.001 --n-epochs 10 --batch-size 64 --validation-size 0.16666