#!/bin/bash
#SBATCH --account=def-xingyu
#SBATCH --time=02:00:00
#SBATCH --mem=10G
#SBATCH --gpus-per-node=v100l:2
#SBATCH --cpus-per-task=4
#SBATCH --array=0-9
module load python/3.8.10
module load cuda/11.4
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r RaPP/requirements.txt
cd RaPP
python src/train.py --target_label $SLURM_ARRAY_TASK_ID --model ae --n_layers 10
python src/train.py --target_label $SLURM_ARRAY_TASK_ID --model vae --n_layers 10
python src/train.py --target_label $SLURM_ARRAY_TASK_ID --model aae --n_layers 10
