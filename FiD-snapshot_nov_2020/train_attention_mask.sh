#测试attention 可视化矩阵， mask_mask代表是否引入mask attention

python -u train_mask.py \
  --train_data_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/FID_Input_DATA/fid_complex2_refine_webqcheckpoint_v2/webqsp_train_test.json \
  --dev_data_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/FID_Input_DATA/fid_normal_mlm100w_multi_webqsp_v2/webqsp_test.json \
  --model_size large \
  --per_gpu_batch_size 1 \
  --n_context 100 \
  --max_passage_length 200 \
  --total_step 100000 \
  --name complex_train+test_1.9 \
  --model_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/FID_checkpoints_ALL/fid_checkpoint_7g/webqsp_fid_t5_large_100passages \
  --checkpoint_dir /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/FID_checkpoints_ALL/fid_20_checkpoints/ \
  --eval_freq 250 \
  # --mask_mask \
  --eval_print_freq 250