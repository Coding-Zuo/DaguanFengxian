#!/bin/bash

# In this example, we show how to train SimCSE using multiple GPU cards and PyTorch's distributed data parallel on supervised NLI dataset.
# Set how many GPUs to use

NUM_GPU=1

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=$(expr $RANDOM + 1000)

# Allow multiple threads
export OMP_NUM_THREADS=1

# Use distributed data parallel
# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"
#python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \

python ../daguan_task/train.py \
    --model_name_or_path /data2/code/DaguanFengxian/pretrain_weight/bert_300 \
    --train_file  /data2/code/DaguanFengxian/bert_model/data/splits/fold_0_nli/nli_for_simcse.csv \
    --output_dir  /data2/code/DaguanFengxian/bert_model/data/outputs/train.bert150k.simcse \
    --num_train_epochs 8 \
    --per_device_train_batch_size 8 \
    --learning_rate 5e-5 \
    --max_seq_length 256 \
    --evaluation_strategy steps \
    --metric_for_best_model avg_transfer \
    --load_best_model_at_end \
    --eval_steps 5000 \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.5 \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --eval_transfer \
    --gradient_checkpointing \
    "$@"
