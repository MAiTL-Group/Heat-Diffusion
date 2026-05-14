#!/usr/bin/env bash
# Sample reconstructions with a pretrained checkpoint.
# Edit configs/fastMRI.yml -> sampling.weight to point to the checkpoint directory.
# Reconstructions are written to ./exp/image_samples/results/.
python main.py --config=fastMRI.yml \
               --exp=./exp \
               --doc=heat \
               --sample \
               --fid \
               --timesteps=50 \
               --eta=1 \
               --image_folder=results
