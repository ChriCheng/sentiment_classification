

nohup python -m src.evaluate \
  --ckpt_path checkpoints/kim_cnn_non_static.pt \
  --test_path data/kim_sst/test.csv \
  --batch_size 64 \
  &> logs/eval_CNN_non_static.log &