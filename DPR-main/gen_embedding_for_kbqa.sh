WEBQSP_DIR="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/UniK-QA-main"
model_dir="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/DPR-main/checkpoint/downloads/checkpoint/retriever/single/nq/hf_bert_base_old.cp"


# for shard_id in {1..16..1} 
# do
# echo "condensed_relation_shard_id : ${shard_id}"
# # //更多请阅读：https://www.yiibai.com/bash/bash-for-loop.html
# 	# condensed_relations: 16
# 	python generate_dense_embeddings_kbqa.py \
# 		model_file="${model_dir}" \
# 		ctx_src="${WEBQSP_DIR}/dpr_inputs/condensed_relations/all_condensed_relations.tsv" \
# 		shard_id=${shard_id} num_shards=16 \
# 		out_file="${WEBQSP_DIR}/models/dpr_predictions/webqsp_condensed_relations/all_condensed_relations.dpr_out"


# echo "condensed_hyper_relations_shard_id : ${shard_id}"
# 	# condensed_hyper_relations: 16
# 	python generate_dense_embeddings_kbqa.py \
# 		model_file="${model_dir}" \
# 		ctx_src="${WEBQSP_DIR}/dpr_inputs/condensed_hyper_relations/all_condensed_hyper_relations.tsv" \
# 		shard_id=${shard_id} num_shards=16 \
# 		out_file="${WEBQSP_DIR}/models/dpr_predictions/webqsp_condensed_hyper_relations/all_condensed_hyper_relations.dpr_out"
# done


# for shard_id in {1..100..1} 
# do
# echo ""
# # 普通relation: 100
# python generate_dense_embeddings_kbqa.py \
# 	model_file="${model_dir}" \
# 	ctx_src="${WEBQSP_DIR}/dpr_inputs/individual_relations/all_relations.tsv" \
# 	shard_id=0 num_shards=100 \
# 	out_file="${WEBQSP_DIR}/models/dpr_predictions/webqsp_relations/all_relations.dpr_out"
# done

python generate_dense_embeddings_kbqa.py \
	model_file="${model_dir}" \
	ctx_src="${WEBQSP_DIR}/dpr_inputs/debug/debug.tsv" \
	shard_id=0 num_shards=16 \
	out_file="${WEBQSP_DIR}/models/dpr_predictions/debug_test/all_condensed_relations.dpr_out"