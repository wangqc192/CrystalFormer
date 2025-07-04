#!/bin/bash
#SBATCH --job-name=crystalformer
#SBATCH --partition=4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH --nodelist=gpu14

hostname
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
echo "GPU_DEVICE_ORDINAL=$GPU_DEVICE_ORDINAL"
nvidia-smi

python -u ./main.py --folder ./data/ --train_path ./data/mp_20/train.csv --valid_path ./data/mp_20/val.csv --batchsize 512 --dropout_rate 0.5 --lamb_w 0.1>out 2>&1
#export TORCH_DISTRIBUTED_TIMEOUT=120
#export TORCH_DISTRIBUTED_INIT_METHOD_TIMEOUT=120

#export HYDRA_FULL_ERROR=1
#srun /public/home/wangqingchang/miniconda3/envs/crystalflow/bin/python /public/home/wangqingchang/DiffCSP/diffcsp/run.py \
#data=alex_mp_ehull data.train_max_epochs=2000 \
#model=flow_polar \
#optim.optimizer.lr=1e-3 \
#optim.optimizer.weight_decay=0 \
#optim.lr_scheduler.factor=0.6 \
#+model.lattice_polar_sigma=0.1 \
#model.decoder.edge_style=knn_frac \
#model.cost_coord=10 model.cost_lattice=1 \
#model.decoder.num_freqs=256 \
#model.decoder.rec_emb=none model.decoder.num_millers=8 \
#+model.decoder.na_emb=0 \
#model.decoder.hidden_dim=512 model.decoder.num_layers=6 \
#train.pl_trainer.devices=3 \
#+train.pl_trainer.strategy=ddp_find_unused_parameters_true \
#logging.wandb.mode=offline \
#logging.wandb.project=crystalflow-gridtest \
#expname=CSP-BS1-LR1-WD1-RF1-K3-LW2-F1-X1-N1-H1-L1-DS11-ES3 \
      #> CSP-BS1-LR1-WD1-RF1-K3-LW2-F1-X1-N1-H1-L1-DS11-ES3.log 2>&1 
