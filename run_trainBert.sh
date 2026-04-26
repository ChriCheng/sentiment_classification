nohup python -m src.bert_train \
  --data_dir preprocessed_file \
  --save_dir checkpoints/bert_best \
  --model_name roberta-base \
  --max_len 64 \
  --batch_size 16 \
  --epochs 4 \
  --lr 2e-5 \
  --weight_decay 1e-2 &> logs/train_bert.log &