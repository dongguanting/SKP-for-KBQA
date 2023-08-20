# export http_proxy=http://10.22.139.49:6666 \
# export https_proxy=http://10.22.139.49:6666 \


# WEBQSP_DIR="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting"
# model_dir="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/DPR_training_checkpoint/complex1_multi_webqsp_mlm/hf_bert_base.cp.2"
# out_dir="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/UniK-QA-main/models/E2E_1216/freebase_rel/"

# for shard_id in {0..10..1} 
# do
# echo "all_relation_shard_id:${shard_id}"
# # 普通relation: 100
# python generate_dense_embeddings_kbqa_hope.py \
# 	model_file="${model_dir}" \
# 	ctx_src="${WEBQSP_DIR}/KBQA_E2E/data/mini_freebase_text.txt" \
# 	shard_id=${shard_id} num_shards=100 \
# 	out_file="${out_dir}"
# done




export http_proxy=http://10.22.139.49:6666 \
export https_proxy=http://10.22.139.49:6666 \


WEBQSP_DIR="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting"
model_dir="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/DPR_training_checkpoint/mlm+cl+drop_1228/hf_bert_base.cp.2"
out_dir="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/UniK-QA-main/models/mlm+cl+drop_122/webqsp_relations/all_relations.dpr_out"

for shard_id in {0..10..1} 
do
echo "all_relation_shard_id:${shard_id}"
# 普通relation: 100
python generate_dense_embeddings_kbqa_hope.py \
	model_file="${model_dir}" \
	ctx_src="${WEBQSP_DIR}/DPR_Input_DATA/dpr_inputs_complex1/individual_relations/all_relations.tsv" \
	shard_id=${shard_id} num_shards=100 \
	out_file="${out_dir}"
done




