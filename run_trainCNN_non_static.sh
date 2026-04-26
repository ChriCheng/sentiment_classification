#!/bin/bash

mkdir -p logs checkpoints

nohup python -m src.train_kim_cnn \
  --data_dir data/kim_sst \
  --save_path checkpoints/kim_cnn_non_static.pt \
  --word2vec_path embeddings/GoogleNews-vectors-negative300.bin \
  --word2vec_binary \
  --max_len 56 \
  --embed_dim 300 \
  --num_filters 100 \
  --kernel_sizes 3 4 5 \
  --dropout 0.5 \
  --batch_size 50 \
  --epochs 25 \
  --lr 1.0 \
  --rho 0.95 \
  --weight_decay 0.0 \
  --max_norm 3.0 \
  --patience 4 \
  &> logs/train_CNN_non_static.log &