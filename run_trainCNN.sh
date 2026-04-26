nohup python -m src.train \
  --data_dir preprocessed_file \
  --save_path checkpoints/textcnn_best.pt \
  --max_len 48 \
  --embed_dim 128 \
  --num_filters 100 \
  --kernel_sizes 3 4 5 \
  --dropout 0.5 \
  --batch_size 64 \
  --epochs 15 \
  --lr 1e-3 \
  --weight_decay 1e-4\
  &> logs/train.log &