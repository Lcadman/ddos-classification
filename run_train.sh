#!/bin/bash
#SBATCH --job-name="DDoS-Classification"          # a name for your job
#SBATCH --partition=kestrel-gpu                   # partition to which job should be submitted
#SBATCH --qos=gpu_short                           # qos type
#SBATCH --nodes=1                                 # node count
#SBATCH --ntasks=3                                # total number of tasks across all nodes
#SBATCH --cpus-per-task=4                         # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=48G                                 # total memory per node
#SBATCH --gres=gpu:3090:3                         # Request 1 GPU
#SBATCH --time=01:00:00                           # total run time limit (HH:MM:SS)

source /s/bach/b/class/cs535/cs535b/ddos-classification/venv/bin/activate
python3 /s/bach/b/class/cs535/cs535b/ddos-classification/example_train.py
