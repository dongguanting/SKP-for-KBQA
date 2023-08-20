WEBQSP_DIR="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting"
model_dir="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/DPR_training_checkpoint/webqsp_cl_0_0.2_10w/hf_bert_base.cp.2"
out_dir="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/UniK-QA-main/models/E2E_1216/test/test_id_qustion"

for shard_id in {0..16..1} 
do
echo "condensed_relation_shard_id : ${shard_id}"
# //更多请阅读：https://www.yiibai.com/bash/bash-for-loop.html
	# condensed_relations: 16
	python generate_dense_embeddings_kbqa_hope.py \
		model_file="${model_dir}" \
		ctx_src="${WEBQSP_DIR}/KBQA_E2E/data/webqsp_test.txt" \
		shard_id=${shard_id} num_shards=16 \
		out_file="${out_dir}"
done


# WEBQSP_DIR="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting"
# model_dir="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/DPR_checkpoint/checkpoint/downloads/checkpoint/retriever/single-adv-hn/nq/hf_bert_base.cp"
# out_dir="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/UniK-QA-main/models/dpr_predictions_adv/webqsp_condensed_relations/all_condensed_relations.dpr_out"

# for shard_id in {0..16..1} 
# do
# echo "condensed_relation_shard_id : ${shard_id}"
# # //更多请阅读：https://www.yiibai.com/bash/bash-for-loop.html
# 	# condensed_relations: 16
# 	python generate_dense_embeddings_kbqa_hope.py \
# 		model_file="${model_dir}" \
# 		ctx_src="${WEBQSP_DIR}/dpr_inputs_v2/condensed_relations/all_condensed_relations.tsv" \
# 		shard_id=${shard_id} num_shards=16 \
# 		out_file="${out_dir}"
# done