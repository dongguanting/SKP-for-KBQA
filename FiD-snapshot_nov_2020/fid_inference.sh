
python test.py \
  --model_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/FID_checkpoints_ALL/fid_20_checkpoints/train+test_1.4/checkpoint/step-3500 \
  --test_data_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/FID_Input_DATA/fid_complex2_refine_webqcheckpoint_v2/webqsp_test.json \
  --model_size large \
  --per_gpu_batch_size 4 \
  --n_context 100 \
  --name mlm_mask_test \
  --checkpoint_dir /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/FiD-snapshot_nov_2020 \
