#!/bin/bash
#SBATCH -J SlurmPythonGPT2
#SBATCH -N 1
#SBATCH --ntasks-per-node=1       # Request 1 cores
#SBATCH --cpus-per-task=2
#SBATCH --mem=160G 
#SBATCH --gres=gpu:H100:1         # Request 1 GPUs
#SBATCH -t 16:00:00
#SBATCH -o Report-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lhuang340@gatech.edu

export HF_HOME=/home/hice1/lhuang340/scratch/hf_cache
export TRANSFORMERS_CACHE=/home/hice1/lhuang340/scratch/hf_cache
export HF_DATASETS_CACHE=/home/hice1/lhuang340/scratch/hf_cache

source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate provenance_env

cd /home/hice1/lhuang340/scratch/research/watermark/FL-Watermark-Radioactivity

srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 7 --dataset_path "./dataset_c4.json" --whether_save_wm --dataset c4 --model_checkpoint "EleutherAI/pythia-410m-deduped" > output_p7_c4_410M_wm.txt &

wait