#!/bin/bash



cd morphologicalpool
python setup.py clean --all install
cd -

python train_unsup.py  --val-image-path='/home/cwj/Data/VesselNN/valid/image.tif'\
                       --val-label-path='/home/cwj/Data/VesselNN/valid/label.tif'\
                       --train-images-path='/home/cwj/Data/VesselNN/train/images'\
                       --train-labels-path='/home/cwj/Data/VesselNN/train/labels'\
                       --batch-size=4\
                       --gpu-ids 0 6 7

# python train_unsup.py  --val-image-path='/data/cwj/DeepVessel/valid/image.png'\
#                        --val-label-path='/data/cwj/DeepVessel/valid/label.png'\
#                        --train-images-path='/data/cwj/DeepVessel/train/images'\
#                        --train-labels-path='/data/cwj/DeepVessel/train/labels'\
#                        --batch-size=8\
#                        --gpu-ids 0 6 7