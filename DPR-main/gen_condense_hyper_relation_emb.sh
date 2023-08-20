WEBQSP_DIR="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting"
model_dir="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/DPR_training_checkpoint/webqsp_cl_0_0.2_10w/hf_bert_base.cp.2"
out_dir="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/UniK-QA-main/models/E2E_1216/train/train_id_qustion"

for shard_id in {0..16..1} 
do
echo "condensed_hyper_relations_shard_id : ${shard_id}"
	# condensed_hyper_relations: 16
	python generate_dense_embeddings_kbqa_hope.py \
		model_file="${model_dir}" \
		ctx_src="${WEBQSP_DIR}/KBQA_E2E/data/webqsp_train.txt" \
		shard_id=${shard_id} num_shards=16 \
		out_file="${out_dir}"
done

# /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/KBQA_E2E/data/webqsp_train.txt
# WEBQSP_DIR="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting"
# model_dir="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/DPR_checkpoint/checkpoint/downloads/checkpoint/retriever/single-adv-hn/nq/hf_bert_base.cp"
# out_dir="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/UniK-QA-main/models/dpr_predictions_adv/webqsp_condensed_hyper_relations/all_condensed_hyper_relations.dpr_out"

# for shard_id in {0..16..1} 
# do
# echo "condensed_hyper_relations_shard_id : ${shard_id}"
# 	# condensed_hyper_relations: 16
# 	python generate_dense_embeddings_kbqa_hope.py \
# 		model_file="${model_dir}" \
# 		ctx_src="${WEBQSP_DIR}/dpr_inputs_v2/condensed_hyper_relations/all_condensed_hyper_relations.tsv" \
# 		shard_id=${shard_id} num_shards=16 \
# 		out_file="${out_dir}"
# done