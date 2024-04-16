#!/bin/bash
#SBATCH --job-name="DDoS-Classification"
#SBATCH --partition=kestrel-gpu
#SBATCH --qos=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=500M
#SBATCH --gres=gpu:3090:2
#SBATCH --time=00:05:00

source /s/bach/b/class/cs535/cs535b/ddos-classification/venv/bin/activate
srun python3 /s/bach/b/class/cs535/cs535b/ddos-classification/example_train.py
