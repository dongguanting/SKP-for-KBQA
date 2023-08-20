export http_proxy=http://10.22.139.49:6666 \
export https_proxy=http://10.22.139.49:6666 \


python /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/DPR-main/dpr/data/download_data.py \
    --resource https://dl.fbaipublicfiles.com/dpr/checkpoint/retriver/single-adv-hn/nq/hf_bert_base.cp \
    --output_dir "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/dongguanting/DPR-main/checkpoint/downloads/checkpoint/retriever/single-adv-hn/nq" \