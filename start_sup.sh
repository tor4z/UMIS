#!/bin/bash


python train_sup.py  --val-image-path='/home/cwj/Data/VesselNN/valid/image.tif'\
                     --val-label-path='/home/cwj/Data/VesselNN/valid/label.tif'\
                     --train-images-path='/home/cwj/Data/VesselNN/train/images'\
                     --train-labels-path='/home/cwj/Data/VesselNN/train/labels'\
                     --gpu-ids 0 6 7