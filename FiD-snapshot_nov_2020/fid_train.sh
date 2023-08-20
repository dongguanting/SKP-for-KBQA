python -u train.py \
  --train_data_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/FID_Input_DATA/fid_normal_mlm100w_multi_webqsp_v2/webqsp_train.json \
  --dev_data_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/FID_Input_DATA/fid_normal_mlm100w_multi_webqsp_v2/webqsp_test.json \
  --model_size large \
  --per_gpu_batch_size 1 \
  --n_context 100 \
  --max_passage_length 200 \
  --total_step 100000 \
  --name mlm100w_multi_webqsp_test1221——hope \
  --model_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/FID_checkpoints_ALL/fid_checkpoint_7g/webqsp_fid_t5_large_100passages \
  --checkpoint_dir /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/FID_checkpoints_ALL/fid_20_checkpoints/ \
  --eval_freq 250 \
  --eval_print_freq 250