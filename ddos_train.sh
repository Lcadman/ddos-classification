#!/bin/bash
#SBATCH --job-name="DDoS-Classification"
#SBATCH --partition=peregrine-gpu
#SBATCH --qos=gpu_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:nvidia_a100_3g.39gb:2
#SBATCH --time=01:00:00

source /s/bach/b/class/cs535/cs535b/ddos-classification/venv/bin/activate
srun python3 /s/bach/b/class/cs535/cs535b/ddos-classification/ddos_binary_classifier.py
