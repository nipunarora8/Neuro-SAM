#!/bin/bash -l
#SBATCH --job-name=Neuro-SAM	# Job name
#SBATCH -o logs/job_%x_%j.log  # Name of stdout output file (%j expands to %jobId, %x expands to %jobName)
#SBATCH --gres=gpu:a100:1 # Total number of GPUs
#SBATCH --partition=a100  # Partition
#SBATCH --time=24:00:00   	# (optional) Maximal run time (hh:mm:ss)

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80


# Load modules
module python/3.12-conda

# Activate the conda environment and go to the directory where the Python script is located
conda activate sam2
cd /home/hpc/iwb3/iwb3102h/samv2/segment-anything-2

# Run python script
python train_dendrites.py --ppn 20 --pnn 10 --batch_size 32 --model_name "small" --logger "True"
# deactivate env
conda deactivate
echo "Neuro-SAM Job finished!!"