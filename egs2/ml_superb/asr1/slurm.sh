#!/bin/bash
#SBATCH --job-name=exp_multi
#SBATCH --partition=speech-gpu
#SBATCH --gpus=1
#SBATCH --constraint="48g"
#SBATCH --cpus-per-task=8
#SBATCH --array=1-5

cd /share/data/speech-lang/users/jcruzado/repos/espnet_og/espnet/egs2/ml_superb/asr1/
eval "$(/share/data/willett-group/jcruzado/miniconda/bin/conda shell.bash hook)"
conda activate espnet_og
./run_multi.sh --stage 11 --expdir exp${SLURM_ARRAY_TASK_ID}
