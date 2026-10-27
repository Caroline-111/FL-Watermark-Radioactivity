#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=20       # Request 20 cores
#SBATCH --cpus-per-task=2
#SBATCH --mem=160G 
#SBATCH --gres=gpu:H100:20         # Request 20 GPUs
#SBATCH -o Report-%j.out

cd PATH/TO/FL-Watermark-Radioactivity

# Table 3
srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 0 --dataset_path "./dataset_6.6%.json" --model_checkpoint "EleutherAI/pythia-160m-deduped" --dirty_worker 2 --wm_size 1024 > p0_c4_6.6%_wm.txt &

srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 1 --dataset_path "./dataset_16.6%.json" --model_checkpoint "EleutherAI/pythia-160m-deduped" --dirty_worker 5 --wm_size 2560 > p1_c4_16.6%_wm.txt &

srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 2 --dataset_path "./dataset_30.0%.json" --model_checkpoint "EleutherAI/pythia-160m-deduped" --dirty_worker 9 --wm_size 4608 > p2_c4_30%_wm.txt &

srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 3 --dataset_path "./dataset_6.6%.json" --model_checkpoint "EleutherAI/pythia-160m-deduped" --dirty_worker 2 --wm_size 1024 --whether_filter > p3_c4_6.6%_filter.txt &

srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 4 --dataset_path "./dataset_16.6%.json" --model_checkpoint "EleutherAI/pythia-160m-deduped" --dirty_worker 5 --wm_size 2560 --whether_filter > p4_c4_16.6%_filter.txt &

srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 5 --dataset_path "./dataset_30.0%.json" --model_checkpoint "EleutherAI/pythia-160m-deduped" --dirty_worker 9 --wm_size 4608 --whether_filter > p5_c4_30%_filter.txt &

# Table 4 
srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 6 --dataset_path "./dataset_KGW_synthetic.json" --model_checkpoint "EleutherAI/pythia-160m-deduped" --whether_synthetic --delta 0 --temperature 0.0 --whether_filter > p6_KGW_synthetic.txt &

srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 7 --dataset_path "./dataset_KGW_natural.json" --model_checkpoint "EleutherAI/pythia-160m-deduped" --delta 0 --temperature 0.0 --whether_filter > p7_KGW_natural.txt &

srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 8 --dataset_path "./dataset_KTH_synthetic.json" --model_checkpoint "EleutherAI/pythia-160m-deduped" --whether_synthetic --watermark kuditipudi --whether_filter > p8_KTH_synthetic.txt &

srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 9 --dataset_path "./dataset_KTH_natural.json" --model_checkpoint "EleutherAI/pythia-160m-deduped" --watermark kuditipudi --whether_filter > p9_KTH_natural.txt &

# Table 5
srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 10 --dataset_path "./dataset_delta_0.json" --model_checkpoint "EleutherAI/pythia-160m-deduped" --whether_synthetic --delta 0 --temperature 0.0 > p10_delta_0_wm.txt &

srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 11 --dataset_path "./dataset_delta_1.json" --model_checkpoint "EleutherAI/pythia-160m-deduped" --whether_synthetic --delta 1 --temperature 0.0 > p11_delta_1_wm.txt &

srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 12 --dataset_path "./dataset_delta_3.json" --model_checkpoint "EleutherAI/pythia-160m-deduped" --whether_synthetic --delta 3 --temperature 0.0 > p12_delta_3_wm.txt &

srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 13 --dataset_path "./dataset_delta_5.json" --model_checkpoint "EleutherAI/pythia-160m-deduped" --whether_synthetic --delta 5 --temperature 0.0 > p13_delta_5_wm.txt &

srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 14 --dataset_path "./dataset_delta_0.json" --model_checkpoint "EleutherAI/pythia-160m-deduped" --whether_synthetic --delta 0 --temperature 0.0 --whether_filter > p14_delta_0_filter.txt &

srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 15 --dataset_path "./dataset_delta_1.json" --model_checkpoint "EleutherAI/pythia-160m-deduped" --whether_synthetic --delta 1 --temperature 0.0 --whether_filter > p15_delta_1_filter.txt &

srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 16 --dataset_path "./dataset_delta_3.json" --model_checkpoint "EleutherAI/pythia-160m-deduped" --whether_synthetic --delta 3 --temperature 0.0 --whether_filter > p16_delta_3_filter.txt &

srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 17 --dataset_path "./dataset_delta_5.json" --model_checkpoint "EleutherAI/pythia-160m-deduped" --whether_synthetic --delta 5 --temperature 0.0 --whether_filter > p17_delta_5_filter.txt &

srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 18 --dataset_path "./dataset_KTH.json" --model_checkpoint "EleutherAI/pythia-160m-deduped" --whether_synthetic --watermark kuditipudi > p18_KTH_wm.txt &

srun --export=ALL --gres=gpu:H100:1 --ntasks=1 --cpus-per-task=2 python train.py --process 19 --dataset_path "./dataset_KTH.json" --model_checkpoint "EleutherAI/pythia-160m-deduped" --whether_synthetic --watermark kuditipudi --whether_filter > p19_KTH_filter.txt &