#!/bin/bash
#SBATCH --job-name=exp_scoring
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --array=1-5

cd /share/data/speech-lang/users/jcruzado/repos/espnet_og/espnet/egs2/ml_superb/asr1/
eval "$(/share/data/willett-group/jcruzado/miniconda/bin/conda shell.bash hook)"
conda activate espnet_og
./run_multi.sh --stage 13 --stop_stage 13 --expdir exp${SLURM_ARRAY_TASK_ID}
