export http_proxy=http://10.22.139.49:6666
export https_proxy=http://10.22.139.49:6666
out="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/DPR_training_checkpoint/DPR_mask_matrix_1229"

python train_dense_encoder_mask.py \
train_datasets="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/biencoder-webquestions-train.json" \
dev_datasets="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/biencoder-webquestions-dev.json" \
train=biencoder_local \
checkpoint_file_name="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/DPR_training_checkpoint/DPR_mask_matrix_1229/hf_bert_base.cp" \
output_dir=${out} \



# export http_proxy=http://10.22.139.49:6666
# export https_proxy=http://10.22.139.49:6666
# out="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/DPR_training_checkpoint/webqsp_mlm100w_no_multi_webq_v3"

# python train_dense_encoder.py \
# train_datasets="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/biencoder-webquestions-train.json" \
# dev_datasets="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/biencoder-webquestions-dev.json" \
# train=biencoder_local \
# output_dir=${out} \

# train_datasets="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/DPR-main_data_process/webqsp_dpr_train_test_delete_empty.json" \