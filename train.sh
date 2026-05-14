#!/usr/bin/env bash
# Train the Heat-Diffusion model on FastMRI.
# Logs and checkpoints will be written to ./exp/logs/heat/<timestamp>/.
python main.py --config=fastMRI.yml --exp=./exp --doc=heat
