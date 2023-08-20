export http_proxy=http://10.22.139.49:6666
export https_proxy=http://10.22.139.49:6666

python test_mask.py \
  --model_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/FID_checkpoints_ALL/fid_checkpoint_7g/webqsp_fid_t5_large_100passages \
  --test_data_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/FID_Input_DATA/fid_normal_mlm100w_multi_webqsp_v2/webqsp_test.json \
  --model_size large \
  --per_gpu_batch_size 2 \
  --n_context 100 \
  --name mlm_mask_test \
  --checkpoint_dir /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/FiD-snapshot_nov_2020 \
  --write_test_results \
  --mask_mask