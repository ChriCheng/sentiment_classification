nohup python -m src.bert_evaluate \
  --model_dir checkpoints/bert_best \
  --test_path preprocessed_file/test.csv \
  --max_len 64 \
  --batch_size 32 \
  &> logs/eval_bert.log &
