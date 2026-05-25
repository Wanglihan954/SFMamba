#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

TRAIN_IMG_SIZE=832
data_cfg_path="configs/data/minima_finetune_${TRAIN_IMG_SIZE}.py"
main_cfg_path="configs/jamma/outdoor/final.py"
pretrained_ckpt="/home/ly/9100pro/Reg/JamMa/jamma_log/jamma/version_10/checkpoints/epoch=21-auc@5=0.637-auc@10=0.770-auc@20=0.862.ckpt"

n_nodes=1
n_gpus_per_node=1
torch_num_workers=4
batch_size=2
pin_memory=true
exp_name="jamma_minima_finetune"

python -u ./train.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --exp_name=${exp_name} \
    --ckpt_path=${pretrained_ckpt} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} --pin_memory=${pin_memory} \
    --check_val_every_n_epoch=1 \
    --log_every_n_steps=200 \
    --flush_logs_every_n_steps=200 \
    --limit_val_batches=1. \
    --num_sanity_val_steps=10 \
    --benchmark=True \
    --max_epochs=30
