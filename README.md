# Bridging the KB-Text Gap: Leveraging Structured Knowledge-aware Pre-training for KBQA


## Overview
This is the repository for our work **SKP**, which is recieved by **CIKM 2023**.

## Brief introduction
We propose a **S**tructured **K**nowledge-aware **P**re-training method (**SKP**). In the pre-training stage, we introduce two novel structured knowledge-aware tasks, guiding the model to effectively learn the implicit relationship and better representations of complex subgraphs. 
In downstream KBQA task, we further design an efficient linearization strategy and an interval attention mechanism, which assist the model to better encode complex subgraphs and shield the interference of irrelevant subgraphs during reasoning respectively.
Detailed experiments and analyses on WebQSP verify the effectiveness of SKP, especially the significant improvement in subgraph retrieval (+4.08% H@10).

## Overall Framework
<img width="733" alt="image" src="https://github.com/dongguanting/Structured-Knowledge-aware-Pretraining-for-KBQA/assets/60767110/c63e55fb-0cee-474c-8dbf-392498ad24e6">

## Quick Start
基于UniKQA框架搭建
https://github.com/facebookresearch/UniK-QA

## Dependencies

- Python 3
- [NumPy](http://www.numpy.org/)
- [PyTorch](http://pytorch.org/) (currently tested on version 1.6.0)
- [Transformers](http://huggingface.co/transformers/) (version 3.0.2, unlikely to work with a different version)

1. 数据处理

```
wget https://dl.fbaipublicfiles.com/UniK-QA/data.tar.xz
tar -xvf data.tar.xz
```

![image-20230414175039929](/Users/dongguanting/Library/Application Support/typora-user-images/image-20230414175039929.png)

准备好以上数据，我们提供两种处理方式
1. 普通线性化

python webqsp_preprocess.py

2. 合并复杂子图的线性化：

python webqsp_preprocess_complex2.py

这些代码均注释详细！最终输出三种tsv数据待编码。

2. DPR预训练：


我们用线性化的子图，对处理好的TSV数据进行结构化知识感知预训练：

首先从预处理好的TSV数据中随机抽取100w条：
bash random_sample_complex1.sh

对于DPR的预训练，我们提供三种模式：
1. 对数据进行MLM+对比学习的预训练：
cd ./bert_pretraining
bash train_mlm_contrastive_mask.sh
2. 只做MLM：
cd ./bert_pretraining
bash train-mlm.sh
3. 只做对比学习：
cd ./bert_pretraining
bash train-contrastive.sh




3. 训练DPR：

​	加载DPR预训练的checkpoint，运行train_encoder1.sh

4. 编码tsv成embedding

使用脚本gen_all_relation_emb1.sh - gen_all_relation_emb10.sh，以及gen_condense_hyper_relation_emb.sh，gen_condense_relation_emb.sh来编码三种tsv

5. 预处理FID的数据

分别运行dpr_inference.py生成DPR output的数据，再用fid_preprocess.py进行进一步筛选，最终生成FID的输入数据

6. 最后，可以使用 UniK-QA 输入训练 FiD 模型。 我们训练有素的 FiD 检查点可以在 [此处](https://dl.fbaipublicfiles.com/UniK-QA/fid_checkpoint.tar.xz) 下载。 （我们的模型是在 2020 年底训练的，因此您可能需要查看旧版本的 FiD。）运行test_origin.sh脚本




## Result

### Main Result
| Model                                                    |  Hits@1    | 
| --------------------- | :------:   | 
| GraftNet                                                 |  69.5     | 
| PullNet                                                    |  68.1    | 
| EMQL                                                      |  75.5     | 
| BERT-KBQA                                                |  72.9    | 
| NSM                                                       |  74.3    | 
| KGT5                                                      | 56.1 | 
| SR-NSM                                                   | 69.5| 
| EmbededKGQA                                               | 72.5| 
| DECAF(Answer only)                                        | 74.7 | 
| UniK-QA∗                                                 | 77.9 |
| SKP (ours)                                               | **79.6** | 


### In-Context Learning Result For LLMs

Since there were very few open source large models when the article was written (2022.12), we now supplement the SKP framework with the results of **In Context Learning** when the LLMs is used as a Reader. Due to the limitation of the Max sequence length of the LLMs, for the **Topk** documents retrieved by the retriever, we select the documents with the highest semantic similarity and truncate them with 2048 tokens as the knowledge prompting for reader (about 5 documents)


| Model                                                    |  Hits@1     | 
| -------------------------------------------------------- | :------:   | 
| SKP(ChatGPT)                                              |  71.9     | 
| SKP(LLAMA)                                                |  coming soon   | 
| SKP(LLAMA2)                                                |  coming soon     | 
| SKP(ChatGLM)                                                |  coming soon     | 
| SKP(ChatGLM2)                                               |  coming soon     | 


**Our Code and Dataset will be released soon!**



