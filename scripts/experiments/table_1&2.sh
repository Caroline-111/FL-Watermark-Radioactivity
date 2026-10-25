#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=16       # Request 16 cores
#SBATCH --cpus-per-task=2
#SBATCH --mem=160G 
#SBATCH --gres=gpu:H100:16         # Request 16 GPUs
#SBATCH -o Report-%j.out

cd PATH/TO/FL-Watermark-Radioactivity

# C4-KGW
srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 0 --dataset_path "./dataset_c4_KGW.json" --dataset c4 --model_checkpoint "EleutherAI/pythia-70m-deduped" --watermark kirchenbauer > p0_c4_70M_KGW_wm.txt &

srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 1 --dataset_path "./dataset_c4_KGW.json" --dataset c4 --model_checkpoint "EleutherAI/pythia-160m-deduped" --watermark kirchenbauer > p1_c4_160M_KGW_wm.txt &

srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 2 --dataset_path "./dataset_c4_KGW.json" --dataset c4 --model_checkpoint "EleutherAI/pythia-410m-deduped" --watermark kirchenbauer > p2_c4_410M_KGW_wm.txt &

# C4-KTH-ITS
srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 3 --dataset_path "./dataset_c4_KTH-ITS.json" --dataset c4 --model_checkpoint "EleutherAI/pythia-70m-deduped" --watermark kuditipudi > p3_c4_70M_KTH-ITS_wm.txt &

srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 4 --dataset_path "./dataset_c4_KTH-ITS.json" --dataset c4 --model_checkpoint "EleutherAI/pythia-160m-deduped" --watermark kuditipudi > p4_c4_160M_KTH-ITS_wm.txt &

srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 5 --dataset_path "./dataset_c4_KTH-ITS.json" --dataset c4 --model_checkpoint "EleutherAI/pythia-410m-deduped" --watermark kuditipudi > p5_c4_410M_KTH-ITS_wm.txt &


# ALPACA-KGW
srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 6 --dataset_path "./dataset_alpaca_KGW.json" --dataset alpaca --model_checkpoint "EleutherAI/pythia-70m-deduped" --watermark kirchenbauer > p6_alpaca_70M_KGW_wm.txt &

srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 7 --dataset_path "./dataset_alpaca_KGW.json" --dataset alpaca --model_checkpoint "EleutherAI/pythia-160m-deduped" --watermark kirchenbauer > p7_alpaca_160M_KGW_wm.txt &

srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 8 --dataset_path "./dataset_alpaca_KGW.json" --dataset alpaca --model_checkpoint "EleutherAI/pythia-410m-deduped" --watermark kirchenbauer > p8_alpaca_410M_KGW_wm.txt &

# ALPACA-KTH-ITS
srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 9 --dataset_path "./dataset_alpaca_KTH-ITS.json" --dataset alpaca --model_checkpoint "EleutherAI/pythia-70m-deduped" --watermark kuditipudi > p9_alpaca_70M_KTH-ITS_wm.txt &

srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 10 --dataset_path "./dataset_alpaca_KTH-ITS.json" --dataset alpaca --model_checkpoint "EleutherAI/pythia-160m-deduped" --watermark kuditipudi > p10_alpaca_160M_KTH-ITS_wm.txt &

srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 11 --dataset_path "./dataset_alpaca_KTH-ITS.json" --dataset alpaca --model_checkpoint "EleutherAI/pythia-410m-deduped" --watermark kuditipudi > p11_alpaca_410M_KTH-ITS_wm.txt &


# Table 2
srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 12 --dataset_path "./dataset_c4_KGW.json" --whether_filter --dataset c4 --model_checkpoint "EleutherAI/pythia-160m-deduped" --watermark kirchenbauer > p12_c4_160M_filter.txt &

srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 13 --dataset_path "./dataset_c4_KGW.json" --whether_filter --dataset c4 --model_checkpoint "EleutherAI/pythia-410m-deduped" --watermark kirchenbauer > p13_c4_410M_filter.txt &

srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 14 --dataset_path "./dataset_alpaca_KGW.json" --whether_filter --dataset alpaca --model_checkpoint "EleutherAI/pythia-160m-deduped" --watermark kirchenbauer > p14_alpaca_160M_filter.txt &

srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 15 --dataset_path "./dataset_alpaca_KGW.json" --whether_filter --dataset alpaca --model_checkpoint "EleutherAI/pythia-410m-deduped" --watermark kirchenbauer > p15_alpaca_410M_filter.txt &

