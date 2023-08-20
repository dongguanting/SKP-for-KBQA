export http_proxy=http://10.22.139.49:6666
export https_proxy=http://10.22.139.49:6666
out="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/DPR_training_checkpoint/webqsp_comlex2_multi+selftrain+test_complex_test"

python train_dense_encoder.py \
train_datasets="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/DPR-main_data_process/complex_100_data/webqsp_dpr_train+test_delete_empty.json" \
dev_datasets="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/DPR-main_data_process/complex_100_test_data/webqsp_dpr_test_delete_empty.json" \
train=biencoder_local \
checkpoint_file_name="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/DPR_checkpoint/checkpoint/downloads/checkpoint/retriever/multi-dataset-dpr/hf_bert_base.cp" \
output_dir=${out} \




# train_datasets="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/DPR-main_data_process/webqsp_dpr_train_test_delete_empty.json" \